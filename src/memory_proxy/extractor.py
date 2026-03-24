from __future__ import annotations

import itertools
import re
from dataclasses import dataclass

from .models import Actor, EventAction, EventType, MemoryEvent, RawMessage
from .zh_semantics import contains_workflow_keyword, is_generic_task_subject


ARTIFACT_PATTERN = re.compile(r"(?:/[\w./-]+)|(?:\b[\w./-]+\.[A-Za-z0-9]+\b)")
VERIFICATION_HINT_RE = re.compile(r"(测试|校验|验证|验收|回归|联调|灰度|复核)", re.IGNORECASE)
FORMAT_INSTRUCTION_RE = re.compile(
    r"^(?:只(?:输出|回答|回复)|不要解释|每条都写成|回答不超过|用两行回答|用一句话|先给结论|再给\d+条理由|最后一次记忆核对|反例检查)",
    re.IGNORECASE,
)
FACT_FIELD_HINT_TOKENS = (
    "代号",
    "负责人",
    "预算",
    "方案",
    "上限",
    "区域",
    "冷备区",
    "入口层",
    "拓扑",
    "规则",
    "目标",
    "时间窗",
    "吞吐",
    "峰值",
    "学段",
    "科目",
    "年级",
    "班级",
    "课程",
    "班主任",
    "批次",
    "产地",
    "风险",
    "等级",
    "时限",
    "保质期",
    "召回",
    "检测",
)
FACT_PREFIX_RE = re.compile(r"^(?:先记住这些事实|再补(?:两条)?事实|当前正式口径|正式口径)[:：]\s*")
DIRECT_FIELD_VALUE_RE = re.compile(
    r"^(?P<field>项目代号|代号|项目负责人|初始回滚负责人|当前回滚负责人|回滚负责人|主区域|欧盟区域|"
    r"节点总上限|节点上限|ID方案|当前正式预算|正式预算|预算|p95目标|回滚时间窗|稳态吞吐|突发峰值|"
    r"学段|科目|年级|班级|班主任|课程方案|课程|批次|产地|风险等级|处理时限|召回级别|保质期|检测项目)"
    r"(?P<value>.+)$"
)
STATE_FACT_PATTERNS = (
    re.compile(r"(?P<field>[\w\u4e00-\u9fff .+-]{2,20}?)从[^，。；;\n]{1,40}?改成(?P<value>[^，。；;\n]+)"),
    re.compile(r"(?P<field>[\w\u4e00-\u9fff .+-]{2,20}?)(?:仍是|还是)(?P<value>[^，。；;\n]+)"),
    re.compile(r"(?P<field>[\w\u4e00-\u9fff .+-]{2,20}?)确定用(?P<value>[^，。；;\n]+)"),
    re.compile(r"(?P<field>[\w\u4e00-\u9fff .+-]{2,20}?)采用(?P<value>.+)$"),
    re.compile(r"(?P<field>[\w\u4e00-\u9fff .+-]{2,20}?)(?:必须|需要|得)(?P<value>[^。；;\n]+)"),
    re.compile(r"(?P<field>[\w\u4e00-\u9fff .+-]{2,20}?)(?:是|为)(?P<value>[^，。；;\n]+)"),
    re.compile(r"(?P<value>[^，。；;\n]{1,40})仍是(?P<field>[\w\u4e00-\u9fff .+-]{2,20})"),
    re.compile(r"(?P<value>[^，。；;\n]{1,40})是(?P<field>[\w\u4e00-\u9fff .+-]{2,20})"),
)


@dataclass(slots=True)
class TriggerMatch:
    event_type: str
    status: str
    action: str
    subject: str
    confidence: float
    details: dict[str, str | int | float | bool]


class RuleBasedExtractor:
    def __init__(self) -> None:
        self._event_counter = itertools.count(1)

    def is_memory_worthy(self, message: RawMessage) -> bool:
        text = message.content.strip()
        lower = text.lower()
        if not text:
            return False
        if message.role == Actor.TOOL:
            return True
        if len(text) <= 3:
            return False
        if ARTIFACT_PATTERN.search(text):
            return True
        if text.endswith(("?", "？")):
            return True
        if contains_workflow_keyword(text):
            return True
        return any(
            token in lower
            for token in (
                "我们现在做",
                "我想",
                "我要",
                "希望",
                "目标",
                "骨架",
                "搭起来",
                "优先",
                "重点",
                "计划",
                "仍是",
                "确定用",
                "采用",
                "预算",
                "负责人",
                "方案",
                "上限",
                "区域",
                "拓扑",
                "时间窗",
                "吞吐",
                "学段",
                "科目",
                "年级",
                "班级",
                "班主任",
                "批次",
                "产地",
                "风险等级",
                "报错",
                "失败",
                "通过",
                "推荐",
                "决定",
                "采用",
                "先用",
                "是否",
                "要不要",
                "need to",
                "must",
                "i want",
                "we want",
                "main goal",
                "focus is",
                "next step",
                "i finished",
                "completed",
                "start with",
                "use ",
                "tests passed",
                "recommend",
            )
        )

    def extract(self, message: RawMessage, session_id: str, turn_id: str) -> list[MemoryEvent]:
        if not self.is_memory_worthy(message):
            return []

        actor = self._normalize_actor(message.role)
        segments = self._split_segments(message.content)
        matches: list[TriggerMatch] = []
        if actor in (Actor.USER, Actor.SYSTEM):
            matches.extend(self._extract_state_updates(message.content))
        for segment in segments:
            matches.extend(self._extract_from_segment(segment, actor))
        matches.extend(self._extract_artifacts(message.content))

        unique: dict[tuple[str, str, str], MemoryEvent] = {}
        for match in matches:
            subject = self._normalize_subject(match.subject)
            if not subject:
                continue
            key = (match.event_type, match.status, subject)
            if key in unique:
                continue
            unique[key] = self._make_event(
                actor=actor,
                session_id=session_id,
                turn_id=turn_id,
                source_message_id=message.message_id,
                event_type=match.event_type,
                action=match.action,
                status=match.status,
                subject=subject,
                confidence=match.confidence,
                details=match.details,
            )
        return list(unique.values())

    def _extract_from_segment(self, segment: str, actor: str) -> list[TriggerMatch]:
        matches: list[TriggerMatch] = []
        text = segment.strip()
        if not text:
            return matches
        if self._is_format_instruction(text):
            return matches

        lower = text.lower()

        if actor in (Actor.USER, Actor.SYSTEM):
            matches.extend(self._extract_user_workflow_updates(text))
            goal = self._match_first(
                text,
                (
                    (r"^(?:我想做(?:一个)?|我想|我要做|我要|希望做|希望|目标是)(.+)$", 0.94),
                    (r"^(?:我们现在做.+?这个[，,]?\s*先把)(.+?)(?:骨架)?(?:搭起来|搭好|做起来|弄起来).*$", 0.92),
                    (r"^(?:先把)(.+?)骨架(?:搭起来|搭好|做起来|弄起来).*$", 0.9),
                    (r"^(?:想做(?:一个)?)(.+)$", 0.9),
                    (r"^(?:i want to build|we want to build|i want to|we want to|i want|we want)(.+)$", 0.92),
                    (r"^(?:the goal is|goal is)(.+)$", 0.88),
                ),
            )
            if goal:
                matches.append(self._build_match(EventType.GOAL, "active", EventAction.ADD, goal[0], goal[1]))

            constraint = self._extract_constraint(text)
            if constraint:
                subject = constraint[0] if len(constraint) == 2 else "".join(constraint)
                matches.append(
                    self._build_match(EventType.CONSTRAINT, "active", EventAction.ADD, subject, 0.88)
                )

            user_task = self._match_first(
                text,
                (
                    (r"^(?:接下来|下一步)(?:还要|还需要|要|得|需要)?(.+)$", 0.89),
                    (r"^(?:还要|还需要|还得|需要继续)(.+)$", 0.86),
                    (r"^(?:next step|next|we still need to|we need to|please)(.+)$", 0.84),
                ),
            )
            if user_task:
                matches.append(
                    self._build_match(
                        EventType.TASK,
                        "proposed",
                        EventAction.ADD,
                        user_task[0],
                        user_task[1],
                    )
                )

            decision = self._extract_user_decision(text)
            if decision:
                matches.append(
                    self._build_match(EventType.DECISION, "active", EventAction.ADD, decision[0], decision[1])
                )

            resolved_decision = self._extract_user_decision_resolution(text)
            if resolved_decision:
                matches.append(
                    self._build_match(
                        EventType.DECISION,
                        "superseded",
                        EventAction.INVALIDATE,
                        resolved_decision[0],
                        resolved_decision[1],
                    )
                )

        if actor == Actor.ASSISTANT:
            verified_subject = self._extract_verified_task_subject(text)
            if verified_subject:
                matches.append(
                    self._build_match(
                        EventType.TASK,
                        "verified_done",
                        EventAction.VERIFY,
                        verified_subject,
                        0.91,
                    )
                )
            done = self._match_first(
                text,
                (
                    (r"^(?:我|我们)?(?:已经|已)?(?:完成了|完成|做完了|做完|搞定了|搞定)(.+)$", 0.94),
                    (r"^(.+?)(?:已经完成|已完成)$", 0.88),
                    (r"^(?:i|we)?(?: have)?(?: already)?(?: finished|completed|implemented|added)(.+)$", 0.9),
                    (r"^(?:i|we) finished(.+)$", 0.88),
                ),
            )
            if done:
                subject = self._normalize_task_subject(done[0])
                if subject:
                    matches.append(
                        self._build_match(EventType.TASK, "claimed_done", EventAction.UPDATE, subject, done[1])
                    )

            plan = self._match_first(
                text,
                (
                    (r"^(?:下一步|接下来|后面)(?:准备|计划|会|将)?(.+)$", 0.93),
                    (r"^(?:准备|计划)(.+)$", 0.88),
                    (r"^(?:next step|next|after that)(.+)$", 0.9),
                    (r"^(?:i will|we will|i am going to|we are going to|plan is to)(.+)$", 0.86),
                ),
            )
            if plan:
                matches.append(self._build_match(EventType.PLAN, "active", EventAction.ADD, plan[0], plan[1]))

            decision = self._match_first(
                text,
                (
                    (r"^(?:我推荐|推荐|我建议|建议|决定|先用|采用|使用)(.+)$", 0.9),
                    (r"^(?:i suggest|suggest|decide to|decided to|start with|use)(.+)$", 0.86),
                ),
            )
            if decision:
                matches.append(
                    self._build_match(EventType.DECISION, "active", EventAction.ADD, decision[0], decision[1])
                )

        if actor == Actor.TOOL:
            matches.append(
                self._build_match(
                    EventType.OBSERVATION,
                    "active",
                    EventAction.ADD,
                    text,
                    0.97,
                    {"kind": self._classify_observation_kind(lower)},
                )
            )
            verified_subject = self._extract_verified_task_subject(text)
            if verified_subject:
                matches.append(
                    self._build_match(
                        EventType.TASK,
                        "verified_done",
                        EventAction.VERIFY,
                        verified_subject,
                        0.88,
                    )
                )
            return matches

        if "报错" in text or "失败" in text or "error" in lower or "failed" in lower:
            matches.append(
                self._build_match(
                    EventType.OBSERVATION,
                    "active",
                    EventAction.ADD,
                    text,
                    0.9,
                    {"kind": self._classify_observation_kind(lower)},
                )
            )

        if "测试通过" in text or "tests passed" in lower:
            matches.append(
                self._build_match(
                    EventType.OBSERVATION,
                    "active",
                    EventAction.ADD,
                    text,
                    0.92,
                    {"kind": "test_result"},
                )
            )

        if actor in (Actor.USER, Actor.SYSTEM) and (
            text.endswith(("?", "？")) or any(token in text for token in ("是否", "要不要", "需不需要"))
        ):
            matches.append(self._build_match(EventType.QUESTION, "open", EventAction.ADD, text, 0.84))

        return matches

    def _extract_artifacts(self, content: str) -> list[TriggerMatch]:
        matches: list[TriggerMatch] = []
        for artifact in ARTIFACT_PATTERN.findall(content):
            if len(artifact) < 4:
                continue
            matches.append(
                self._build_match(EventType.ARTIFACT, "active", EventAction.ADD, artifact, 0.82)
            )
        return matches

    def _build_match(
        self,
        event_type: str,
        status: str,
        action: str,
        subject: str,
        confidence: float,
        details: dict[str, str | int | float | bool] | None = None,
    ) -> TriggerMatch:
        return TriggerMatch(
            event_type=event_type,
            status=status,
            action=action,
            subject=subject,
            confidence=confidence,
            details=details or {},
        )

    def _make_event(
        self,
        actor: str,
        session_id: str,
        turn_id: str,
        source_message_id: str,
        event_type: str,
        action: str,
        status: str,
        subject: str,
        confidence: float,
        details: dict[str, str | int | float | bool],
    ) -> MemoryEvent:
        event_id = f"evt_{next(self._event_counter):04d}"
        return MemoryEvent(
            event_id=event_id,
            session_id=session_id,
            turn_id=turn_id,
            source_message_ids=[source_message_id],
            actor=actor,
            type=event_type,
            action=action,
            status=status,
            subject=subject,
            confidence=confidence,
            details=details,
        )

    def _split_segments(self, content: str) -> list[str]:
        return [segment.strip() for segment in re.split(r"[，。；;\n]+", content) if segment.strip()]

    def _normalize_actor(self, role: str) -> str:
        role = role.lower()
        if role in {Actor.USER, Actor.ASSISTANT, Actor.TOOL, Actor.SYSTEM}:
            return role
        return Actor.USER

    def _normalize_subject(self, subject: str) -> str:
        subject = subject.strip().strip("`'\" ")
        subject = re.sub(r"^[：:=-]\s*", "", subject)
        subject = re.sub(r"\s+", " ", subject)
        return subject

    def _normalize_task_subject(self, subject: str) -> str:
        normalized = self._normalize_subject(subject)
        normalized = re.sub(r"^(?:你帮我|帮我|你)\s*", "", normalized)
        normalized = re.sub(r"^把", "", normalized)
        normalized = re.sub(r"^(?:那个|这个|这件|这项|该)\s*", "", normalized)
        if not re.match(r"^创建.+任务$", normalized):
            normalized = re.sub(r"\s*(?:工作|事情|任务)$", "", normalized)
        normalized = normalized.strip(" ，。；;")
        if self._is_generic_task_subject(normalized):
            return ""
        return normalized

    def _is_generic_task_subject(self, subject: str) -> bool:
        return is_generic_task_subject(subject)

    def _normalize_verified_task_subject(self, subject: str) -> str:
        normalized = self._normalize_task_subject(subject)
        normalized = re.sub(r"(?:已经|已)(?:测试|校验|验证|验收|回归|联调|灰度|复核)$", "", normalized)
        return normalized.strip(" ，。；;")

    def _looks_like_verification_task(self, subject: str) -> bool:
        normalized = self._normalize_verified_task_subject(subject)
        if not normalized:
            return False
        if normalized.startswith(("灰度", "联调", "回归", "验收", "复核")):
            return True
        return bool(VERIFICATION_HINT_RE.search(normalized))

    def _extract_user_workflow_updates(self, text: str) -> list[TriggerMatch]:
        matches: list[TriggerMatch] = []

        verified_subject = self._extract_verified_task_subject(text)
        if verified_subject:
            matches.append(
                self._build_match(
                    EventType.TASK,
                    "verified_done",
                    EventAction.VERIFY,
                    verified_subject,
                    0.93,
                )
            )

        done = self._match_first(
            text,
            (
                (r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)(?:完成了|完成|做完了|做完|搞定了|搞定)$", 0.93),
                (r"^(.+?)(?:完成了|做完了|搞定了)$", 0.88),
            ),
        )
        if done:
            subject = self._normalize_task_subject(done[0])
            if subject:
                matches.append(
                    self._build_match(EventType.TASK, "claimed_done", EventAction.UPDATE, subject, done[1])
                )

        todo = self._match_first(
            text,
            (
                (
                    r"^(?:请|麻烦|烦请)?(?:你帮我|你先|帮我|你)(?:先|继续|再)?(?:把)?(.+?)(?:干了|做了|搞了|处理了|处理掉|完成|做完|搞定)(?:吧)?$",
                    0.94,
                ),
                (
                    r"^(?:请|麻烦|烦请)?(?:你帮我|你先|帮我|你)(?:先|继续|再)?(创建.+?)(?:吧)?$",
                    0.93,
                ),
            ),
        )
        if todo:
            subject = self._normalize_task_subject(todo[0])
            if subject:
                matches.append(
                    self._build_match(EventType.TASK, "proposed", EventAction.ADD, subject, todo[1])
                )

        paused = self._match_first(
            text,
            (
                (r"^(?:先|暂时)(?:把)?(.+?)(?:放一下|搁置|暂停)(?:吧)?$", 0.9),
                (r"^(?:把)?(.+?)(?:先放一下|先放一放|先搁置|先暂停|暂停一下)(?:吧)?$", 0.9),
            ),
        )
        if paused:
            subject = self._normalize_task_subject(paused[0])
            if subject:
                matches.append(
                    self._build_match(EventType.TASK, "resolved", EventAction.RESOLVE, subject, paused[1])
                )

        return matches

    def _extract_user_decision(self, text: str) -> tuple[str, float] | None:
        matched = re.match(r"^(?:还是|仍然|继续|就|先|那就)按(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_decision_subject(matched.group(1), prefix="按")
            if subject:
                return subject, 0.9

        matched = re.match(r"^(?:还是|仍然|继续|就|先|那就)用(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_decision_subject(matched.group(1), prefix="用")
            if subject:
                return subject, 0.86

        matched = re.match(r"^(?:改用)(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_decision_subject(matched.group(1), prefix="改用")
            if subject:
                return subject, 0.88

        matched = re.match(r"^(?:改成|换成)(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_decision_subject(matched.group(1))
            if subject:
                return subject, 0.88
        return None

    def _extract_user_decision_resolution(self, text: str) -> tuple[str, float] | None:
        matched = re.match(r"^(?:不要|别)(?:再)?回到(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_subject(matched.group(1))
            if subject:
                return subject, 0.9

        matched = re.match(r"^(?:不要|别)(?:再)?按(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_subject(matched.group(1))
            if subject:
                return f"按{subject}", 0.86

        matched = re.match(r"^(?:不要|别)(?:再)?用(.+)$", text, re.IGNORECASE)
        if matched:
            subject = self._normalize_subject(matched.group(1))
            if subject:
                return f"用{subject}", 0.84
        return None

    def _normalize_decision_subject(self, raw: str, *, prefix: str | None = None) -> str:
        subject = self._normalize_subject(raw)
        subject = re.sub(r"(?:吧|呀|啊|哦|喔|嘛|呗)$", "", subject).strip()
        subject = re.sub(r"走$", "", subject).strip()
        if not subject:
            return ""

        if prefix == "按":
            subject = re.sub(r"^按", "", subject).strip()
            return f"按{subject}" if subject else ""
        if prefix == "用":
            subject = re.sub(r"^用", "", subject).strip()
            return f"用{subject}" if subject else ""
        if prefix == "改用":
            subject = re.sub(r"^(?:改用|用)", "", subject).strip()
            return f"改用{subject}" if subject else ""
        return subject

    def _extract_state_updates(self, content: str) -> list[TriggerMatch]:
        matches: list[TriggerMatch] = []
        seen: set[str] = set()
        clauses = [segment.strip() for segment in re.split(r"[；;\n。]+", content) if segment.strip()]
        for clause in clauses:
            normalized_clause = FACT_PREFIX_RE.sub("", clause).strip()
            if not normalized_clause or self._is_format_instruction(normalized_clause):
                continue
            if self._looks_like_decision_clause(normalized_clause):
                continue
            direct = DIRECT_FIELD_VALUE_RE.match(normalized_clause)
            if direct:
                self._append_state_match(
                    matches,
                    seen,
                    direct.group("field"),
                    direct.group("value"),
                    confidence=0.94,
                )
            for pattern in STATE_FACT_PATTERNS:
                for matched in pattern.finditer(normalized_clause):
                    field = matched.groupdict().get("field", "")
                    value = matched.groupdict().get("value", "")
                    self._append_state_match(matches, seen, field, value, confidence=0.9)
        return matches

    def _append_state_match(
        self,
        matches: list[TriggerMatch],
        seen: set[str],
        raw_field: str,
        raw_value: str,
        *,
        confidence: float,
    ) -> None:
        slot = self._normalize_state_slot(raw_field)
        value = self._normalize_state_value(raw_value)
        if not slot or not value:
            return
        key = f"{slot}={value}"
        if key in seen:
            return
        seen.add(key)
        matches.append(
            self._build_match(
                EventType.STATE,
                "active",
                EventAction.ADD,
                key,
                confidence,
                {"slot": slot, "value": value},
            )
        )

    def _normalize_state_slot(self, raw_field: str) -> str:
        field = self._normalize_subject(raw_field)
        field = field.strip("“”‘’")
        field = re.sub(r"^(?:当前|目前|任何变更|任何)\s*", "", field)
        field = re.sub(r"\s+", "", field)
        if not field:
            return ""

        aliases = {
            "项目代号": "代号",
            "回滚负责人": "当前回滚负责人",
            "正式预算": "当前正式预算",
            "预算": "当前正式预算",
            "节点上限": "节点总上限",
            "欧盟数据": "欧盟数据规则",
            "国内灰度日志": "国内灰度日志规则",
        }
        if field in aliases:
            field = aliases[field]

        if field.startswith("Postgres"):
            return "Postgres拓扑"
        if "JetStream" in field:
            return "JetStream拓扑"
        if field in {"当前正式口径", "正式口径"}:
            return ""
        if self._looks_like_state_field(field):
            return field
        return ""

    def _normalize_state_value(self, raw_value: str) -> str:
        value = self._normalize_subject(raw_value)
        value = value.strip("“”‘’")
        value = re.sub(r"^(?:是|为|仍是)\s*", "", value)
        value = re.sub(r"(?:[,，]\s*只回复[:：]?.*|[,，]\s*只输出[:：]?.*|[,，]\s*只回答[:：]?.*)$", "", value)
        value = re.sub(r"\s+", " ", value).strip(" ，。；;")
        if not value or self._is_format_instruction(value):
            return ""
        return value

    def _looks_like_state_field(self, field: str) -> bool:
        if not field or len(field) > 20:
            return False
        if self._is_format_instruction(field):
            return False
        if any(
            token in field
            for token in ("如果", "请", "回答", "输出", "解释", "反例", "核对", "有人说", "有人建议", "建议", "批准")
        ):
            return False
        return any(token in field for token in FACT_FIELD_HINT_TOKENS)

    def _is_format_instruction(self, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False
        return bool(FORMAT_INSTRUCTION_RE.match(stripped))

    def _looks_like_decision_clause(self, text: str) -> bool:
        stripped = text.strip()
        return bool(
            re.match(
                r"^(?:还是|仍然|继续|就|先|改用|不要|别|收到(?:，|,)?(?:后续)?|收到(?:，|,)?继续)(?:再)?(?:按|用|回到)",
                stripped,
                re.IGNORECASE,
            )
        )

    def _extract_constraint(self, text: str) -> tuple[str, float] | None:
        preserved_negative = self._match_first(
            text,
            (
                (r"^((?:(?:尽可能|尽量|最好|务必|必须|需要|得)\s*)*(?:不要|不能).+)$", 0.94),
                (
                    r"^((?:(?:must|should|need to|please)\s+)*(?:must not|should not|need not|do not|don't|cannot|can't)\s+.+)$",
                    0.92,
                ),
            ),
        )
        if preserved_negative:
            return preserved_negative

        return self._match_first(
            text,
            (
                (r"^(?:必须|需要|得|优先|重点是|核心目标是)(.+)$", 0.93),
                (r"^(?:并且|而且|但|但是|不过)(.+)$", 0.82),
                (r"^(.+?)(?:必须|需要优先|优先考虑)(.+)$", 0.82),
                (r"^(?:must|need to|should|priority is|focus is)(.+)$", 0.9),
                (r"^(?:the main goal is)(.+)$", 0.88),
            ),
        )

    def _match_first(self, text: str, patterns: tuple[tuple[str, float], ...]) -> tuple[str, float] | None:
        for pattern, confidence in patterns:
            matched = re.match(pattern, text, re.IGNORECASE)
            if matched:
                groups = [group.strip(" ：:-") for group in matched.groups() if group and group.strip(" ：:-")]
                if not groups:
                    return None
                subject = " ".join(groups)
                return subject, confidence
        return None

    def _classify_observation_kind(self, lower_text: str) -> str:
        if "pass" in lower_text or "通过" in lower_text:
            return "test_result"
        if "error" in lower_text or "报错" in lower_text:
            return "error"
        if "fail" in lower_text or "失败" in lower_text:
            return "failure"
        return "note"

    def _extract_verified_task_subject(self, text: str) -> str | None:
        patterns = (
            r"tests? passed(?: for)? (.+)$",
            r"测试通过(?:了)?[:：]?\s*(.+)$",
            r"(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核)(?:通过|完成)(?:了)?[:：]?\s*(.+)$",
            r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)?通过(?:了)?(?:测试|校验|验证|验收|回归|联调|灰度|巡检|复核)$",
        )
        for pattern in patterns:
            matched = re.search(pattern, text, re.IGNORECASE)
            if matched:
                subject = self._normalize_verified_task_subject(matched.group(1))
                if subject:
                    return subject
        for pattern, require_hint in (
            (r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)?(?:通过了|通过)$", False),
            (r"^(?:现在|目前|刚刚|刚才)?(.+?)(?:已经|已)?(?:完成了|完成)$", True),
        ):
            matched = re.search(pattern, text, re.IGNORECASE)
            if not matched:
                continue
            subject = self._normalize_verified_task_subject(matched.group(1))
            if not subject:
                continue
            if require_hint and not self._looks_like_verification_task(subject):
                continue
            if not require_hint and not (self._looks_like_verification_task(subject) or "通过" in text):
                continue
            return subject
        return None
