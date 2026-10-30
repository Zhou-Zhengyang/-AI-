import type { BankState, ToolResult } from "../../shared/state.ts";
import { executeTool, yuanText } from "../../shared/state.ts";
import type { AuditChain, Decision, Intent, Policy, Ticket } from "../../shared/rules.ts";
import { dailyStats, evaluate, extractAmount, extractPayee, issueTicket, parseIntent, redact, specOf, verifyTicket } from "../../shared/rules.ts";
import type { ChatMessage, Transport } from "./transport.ts";

/** 钩子：pi 三段式的第三段。调用前后各一个，用来插拦截与记账。 */
export interface AgentHooks {
  /** 工具执行前：返回 allow 为假即拒绝执行。 */
  beforeTool?: (input: {
    tool: string;
    slots: Record<string, unknown>;
    decision: Decision;
  }) => { allow: boolean; reason?: string };
  /** 工具执行后：记账、埋点、回调。 */
  afterTool?: (input: { tool: string; slots: Record<string, unknown>; result: ToolResult }) => void;
}

export interface AgentOptions {
  state: BankState;
  transport: Transport;
  hooks?: AgentHooks;
  chain: AuditChain;
  policy?: Policy & { confirmTtlSeconds: number };
  /** 思考模式下才调模型；关闭时只走规则，省 token 也省时间。 */
  useModel?: boolean;
}

export interface ChatReply {
  sessionId: string;
  reply: string;
  scene: string;
  decision: Decision;
  confirmation?: { token: string; expiresAt: number };
  execution?: ToolResult;
  audit: { seq: number; hash: string };
  missing: string[];
  intentSource: "rule" | "model";
}

/** 注入特征：命中即拦，不做语义判断。 */
const INJECTION = /(忽略.{0,4}(之前|以上|此前).{0,4}(指令|要求)|ignore\s+previous|你现在是|扮演|开发者模式|输出.{0,4}提示词|system\s*prompt)/i;

/** 归一化：全角转半角、去零宽字符。 */
function normalize(text: string): string {
  return text
    .replace(/[\uff01-\uff5e]/g, (char) => String.fromCharCode(char.charCodeAt(0) - 0xfee0))
    .replace(/[\u200b-\u200f\u2028\u2029\ufeff]/g, "")
    .trim();
}

function slotAmount(slots: Record<string, unknown>): number | undefined {
  return typeof slots.amountCents === "number" ? (slots.amountCents as number) : undefined;
}

function slotPayee(slots: Record<string, unknown>): string | undefined {
  return typeof slots.payee === "string" && slots.payee !== "" ? (slots.payee as string) : undefined;
}

/** 模型兜底用的工具契约：给出真实字段名与类型，模型才知道怎么填槽位。 */
const TOOL_SCHEMAS: Record<string, { description: string; parameters: Record<string, unknown> }> = {
  query_balance: { description: "查询账户余额", parameters: { type: "object", properties: {}, required: [] } },
  analyze_bills: { description: "账单统计与异常", parameters: { type: "object", properties: {}, required: [] } },
  transfer: {
    description: "向收款人转账",
    parameters: {
      type: "object",
      properties: {
        payee: { type: "string", description: "收款人姓名" },
        amountCents: { type: "integer", description: "转账金额，单位分，正整数" },
      },
      required: ["payee", "amountCents"],
    },
  },
  list_subscriptions: { description: "列出订阅与下次扣费", parameters: { type: "object", properties: {}, required: [] } },
  cancel_subscription: {
    description: "取消订阅",
    parameters: {
      type: "object",
      properties: { subscriptionName: { type: "string", description: "订阅名称" } },
      required: ["subscriptionName"],
    },
  },
  lock_card: {
    description: "锁定或解锁卡片",
    parameters: { type: "object", properties: { cardId: { type: "string", description: "卡片编号" } }, required: [] },
  },
  report_loss: {
    description: "挂失卡片，不可撤销",
    parameters: { type: "object", properties: { cardId: { type: "string", description: "卡片编号" } }, required: [] },
  },
  wealth_redeem: {
    description: "赎回理财",
    parameters: { type: "object", properties: { amountCents: { type: "integer", description: "赎回金额，单位分" } }, required: [] },
  },
  birthday_lock: {
    description: "锁定资金准备生日礼遇",
    parameters: { type: "object", properties: { amountCents: { type: "integer", description: "锁定金额，单位分" } }, required: [] },
  },
  unknown: { description: "无法判断用户意图时调用", parameters: { type: "object", properties: {}, required: [] } },
};

export class Agent {
  private readonly options: AgentOptions;
  private readonly pending = new Map<string, Ticket & { sessionId: string }>();
  private readonly drafts = new Map<string, Intent>();

  constructor(options: AgentOptions) {
    this.options = options;
  }

  get state(): BankState {
    return this.options.state;
  }

  get transportInfo(): { model: string; available: boolean } {
    return { model: this.options.transport.model, available: this.options.transport.available };
  }

  /** 模型兜底：只产出意图，输出必须过白名单与参数校验。 */
  private async askModel(text: string): Promise<Intent | null> {
    if (!this.options.useModel || !this.options.transport.available) return null;
    const tools = Object.entries(TOOL_SCHEMAS).map(([name, schema]) => ({
      name,
      description: schema.description,
      parameters: schema.parameters,
    }));
    const messages: ChatMessage[] = [
      { role: "system", content: "你是银行智能体的意图解析器。只调用一个工具，金额一律用 amountCents 整数分表示。判断不了调用 unknown。" },
      { role: "user", content: text },
    ];
    try {
      const reply = await this.options.transport.complete(messages, tools);
      const call = reply.toolCalls[0];
      if (call === undefined || call.name === "unknown") return null;
      const spec = specOf(call.name);
      if (spec === undefined) return null;
      let slots: Record<string, unknown> = {};
      try {
        const parsed = JSON.parse(call.arguments) as unknown;
        if (parsed !== null && typeof parsed === "object" && !Array.isArray(parsed)) slots = parsed as Record<string, unknown>;
      } catch {
        return null;
      }
      const missing: string[] = [];
      if (call.name === "transfer") {
        if (typeof slots.payee !== "string" || slots.payee === "") missing.push("payee");
        if (typeof slots.amountCents !== "number" || !Number.isInteger(slots.amountCents) || slots.amountCents <= 0) missing.push("amountCents");
      } else if (call.name === "cancel_subscription") {
        if (typeof slots.subscriptionName !== "string" || slots.subscriptionName === "") missing.push("subscriptionName");
      }
      return { scene: spec.scene, tool: call.name, slots, missing, confidence: 0.9 };
    } catch {
      return null;
    }
  }

  async chat(sessionId: string, raw: string): Promise<ChatReply> {
    const text = normalize(raw);
    const policy = this.options.policy ?? { singleLimitCents: 100000, dailyLimitCents: 500000, firstPayeeLimitCents: 50000, maxDailyCount: 5, confirmTtlSeconds: 120 };

    if (INJECTION.test(text)) {
      const decision: Decision = {
        level: "red",
        blocked: true,
        requiresConfirm: false,
        hits: [{ rule: "injection", detail: "命中指令覆盖或角色扮演特征" }],
        summary: "输入含注入特征",
      };
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: "none",
        decision: "blocked_injection",
        level: "red",
        rules: ["injection"],
        result: "已拦截",
      });
      return {
        sessionId,
        reply: "这句话里包含改变规则的内容，我不会执行。请直接说要办什么业务。",
        scene: "unknown",
        decision,
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource: "rule",
      };
    }

    let intent = parseIntent(text, this.options.state);
    let intentSource: "rule" | "model" = "rule";
    const draft = this.drafts.get(sessionId);
    if (intent.scene === "unknown" && draft !== undefined) {
      const followAmount = extractAmount(text);
      const followPayee = extractPayee(text);
      const slots: Record<string, unknown> = { ...draft.slots };
      if (followAmount !== null) slots.amountCents = followAmount;
      if (followPayee !== null) slots.payee = followPayee;
      const missing: string[] = [];
      if (draft.tool === "transfer") {
        if (slots.amountCents === undefined) missing.push("amountCents");
        if (slots.payee === undefined) missing.push("payee");
      }
      if (missing.length === 0) {
        intent = { ...draft, slots, missing, confidence: 0.95 };
      }
    }
    if (intent.reject !== undefined) {
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: intent.tool,
        decision: "rejected_intent",
        level: "red",
        rules: [intent.reject.rule],
        result: intent.reject.detail,
      });
      return {
        sessionId,
        reply: `这个操作我不能执行：${intent.reject.detail}。`,
        scene: intent.scene,
        decision: { level: "red", blocked: true, requiresConfirm: false, hits: [{ rule: intent.reject.rule, detail: intent.reject.detail }], summary: "解析层拒绝" },
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource: "rule",
      };
    }
    if (intent.scene === "unknown" || intent.confidence < 0.7) {
      const fromModel = await this.askModel(text);
      if (fromModel !== null) {
        intent = fromModel;
        intentSource = "model";
      }
    }

    if (intent.scene === "unknown") {
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: "unknown",
        decision: "unmatched",
        level: "unknown",
        rules: ["intent-miss"],
        result: "未识别",
      });
      return {
        sessionId,
        reply: "我没有听懂。可以试：查余额、分析账单、转给小王 500 元、取消云幕视频会员、锁定卡片、赎回理财。",
        scene: "unknown",
        decision: { level: "red", blocked: true, requiresConfirm: false, hits: [{ rule: "intent-miss", detail: "未识别意图" }], summary: "未识别" },
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource,
      };
    }

    if (intent.missing.length > 0) {
      this.drafts.set(sessionId, intent);
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: intent.tool,
        decision: "need_slots",
        level: "unknown",
        rules: ["slot-missing"],
        result: `缺失 ${intent.missing.join(",")}`,
      });
      return {
        sessionId,
        reply: `还需要确认：${intent.missing.join("、")}。`,
        scene: intent.scene,
        decision: { level: "green", blocked: false, requiresConfirm: false, hits: [{ rule: "slot-missing", detail: intent.missing.join("、") }], summary: "等待补充参数" },
        audit: { seq: record.seq, hash: record.hash },
        missing: intent.missing,
        intentSource,
      };
    }

    this.drafts.delete(sessionId);
    const decision = evaluate(intent, policy, dailyStats(this.options.chain));
    const blocked = this.options.hooks?.beforeTool?.({ tool: intent.tool, slots: intent.slots, decision });
    if (blocked !== undefined && !blocked.allow) {
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: intent.tool,
        decision: "blocked_hook",
        level: "red",
        rules: ["hook"],
        result: blocked.reason ?? "钩子拒绝",
      });
      return {
        sessionId,
        reply: `这个操作被拦截：${blocked.reason ?? "不符合规则"}。`,
        scene: intent.scene,
        decision: { ...decision, level: "red", blocked: true },
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource,
      };
    }

    if (decision.blocked) {
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: intent.tool,
        decision: "blocked_policy",
        level: "red",
        rules: decision.hits.map((hit) => hit.rule),
        result: decision.summary,
      });
      return {
        sessionId,
        reply: `这个操作我不能直接执行。${decision.hits.filter((hit) => hit.rule !== "base-level").map((hit) => hit.detail).join("；")}。可以在手机上完成人脸验证后办理。`,
        scene: intent.scene,
        decision,
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource,
      };
    }

    if (decision.requiresConfirm) {
      const ticket = issueTicket(sessionId, intent.tool, intent.slots, policy.confirmTtlSeconds);
      this.pending.set(sessionId, ticket);
      const record = this.options.chain.append({
        sessionId,
        userText: redact(text),
        tool: intent.tool,
        decision: "pending_confirm",
        level: "yellow",
        rules: decision.hits.map((hit) => hit.rule),
        result: `等待确认：${decision.summary}`,
      });
      return {
        sessionId,
        reply: `准备执行：${decision.summary}。需要你确认，${policy.confirmTtlSeconds} 秒内有效。`,
        scene: intent.scene,
        decision,
        confirmation: { token: ticket.token, expiresAt: ticket.expiresAt },
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource,
      };
    }

    const execution = executeTool(intent.tool, intent.slots, this.options.state);
    this.options.hooks?.afterTool?.({ tool: intent.tool, slots: intent.slots, result: execution });
    const record = this.options.chain.append({
      sessionId,
      userText: redact(text),
      tool: intent.tool,
      decision: execution.ok ? "auto_execute" : "executed_with_error",
      level: "green",
      rules: decision.hits.map((hit) => hit.rule),
      result: execution.ok ? execution.message : (execution.error ?? "执行失败"),
      amountCents: slotAmount(intent.slots),
      payee: slotPayee(intent.slots),
    });
    return {
      sessionId,
      reply: execution.message,
      scene: intent.scene,
      decision,
      execution,
      audit: { seq: record.seq, hash: record.hash },
      missing: [],
      intentSource,
    };
  }

  async confirm(sessionId: string, token: string): Promise<ChatReply> {
    const ticket = this.pending.get(sessionId);
    const check = verifyTicket(ticket, sessionId, token);
    if (ticket === undefined || !check.ok) {
      const record = this.options.chain.append({
        sessionId,
        userText: "确认",
        tool: ticket?.tool ?? "unknown",
        decision: "blocked_token_invalid",
        level: "red",
        rules: ["confirm-token"],
        result: check.reason,
      });
      return {
        sessionId,
        reply: `确认失败：${check.reason}。操作没有执行。`,
        scene: "unknown",
        decision: { level: "red", blocked: true, requiresConfirm: false, hits: [{ rule: "confirm-token", detail: check.reason }], summary: "令牌校验失败" },
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource: "rule",
      };
    }
    const policy = this.options.policy ?? { singleLimitCents: 100000, dailyLimitCents: 500000, firstPayeeLimitCents: 50000, maxDailyCount: 5, confirmTtlSeconds: 120 };
    const reIntent: Intent = { scene: specOf(ticket.tool)?.scene ?? "unknown", tool: ticket.tool, slots: ticket.slots, missing: [], confidence: 1 };
    const reDecision = evaluate(reIntent, policy, dailyStats(this.options.chain));
    const reBlocked = this.options.hooks?.beforeTool?.({ tool: ticket.tool, slots: ticket.slots, decision: reDecision });
    if (reDecision.blocked || (reBlocked !== undefined && !reBlocked.allow)) {
      this.pending.delete(sessionId);
      const reason = reDecision.blocked
        ? reDecision.hits.filter((hit) => hit.rule !== "base-level").map((hit) => hit.detail).join("；")
        : (reBlocked?.reason ?? "钩子拒绝");
      const record = this.options.chain.append({
        sessionId,
        userText: "确认",
        tool: ticket.tool,
        decision: "blocked_confirm",
        level: "red",
        rules: reDecision.blocked ? reDecision.hits.map((hit) => hit.rule) : ["hook"],
        result: reason,
      });
      return {
        sessionId,
        reply: `确认时重新判定，不能执行：${reason}。操作没有执行。`,
        scene: specOf(ticket.tool)?.scene ?? "unknown",
        decision: { ...reDecision, level: "red", blocked: true, requiresConfirm: false },
        audit: { seq: record.seq, hash: record.hash },
        missing: [],
        intentSource: "rule",
      };
    }
    this.pending.delete(sessionId);
    const execution = executeTool(ticket.tool, ticket.slots, this.options.state);
    this.options.hooks?.afterTool?.({ tool: ticket.tool, slots: ticket.slots, result: execution });
    const record = this.options.chain.append({
      sessionId,
      userText: "确认",
      tool: ticket.tool,
      decision: execution.ok ? "confirmed" : "confirmed_failed",
      level: "yellow",
      rules: ["confirm-token"],
      result: execution.ok ? execution.message : (execution.error ?? "执行失败"),
      amountCents: slotAmount(ticket.slots),
      payee: slotPayee(ticket.slots),
    });
    return {
      sessionId,
      reply: execution.message === "" ? (execution.error ?? "执行失败") : execution.message,
      scene: specOf(ticket.tool)?.scene ?? "unknown",
      decision: { level: "yellow", blocked: false, requiresConfirm: false, hits: [{ rule: "confirmed", detail: "用户已确认" }], summary: "已确认执行" },
      execution,
      audit: { seq: record.seq, hash: record.hash },
      missing: [],
      intentSource: "rule",
    };
  }

  cancel(sessionId: string): ChatReply {
    this.pending.delete(sessionId);
    this.drafts.delete(sessionId);
    const record = this.options.chain.append({
      sessionId,
      userText: "取消",
      tool: "none",
      decision: "cancelled",
      level: "green",
      rules: ["cancel"],
      result: "已取消待确认操作",
    });
    return {
      sessionId,
      reply: "已取消，没有执行任何操作。",
      scene: "unknown",
      decision: { level: "green", blocked: false, requiresConfirm: false, hits: [{ rule: "cancel", detail: "用户取消" }], summary: "已取消" },
      audit: { seq: record.seq, hash: record.hash },
      missing: [],
      intentSource: "rule",
    };
  }

  snapshot(): Record<string, unknown> {
    const state = this.options.state;
    return {
      account: { holder: state.account.holder, balanceYuan: yuanText(state.account.balance), availableYuan: yuanText(state.account.available) },
      cards: state.cards,
      subscriptions: state.subscriptions,
      holdingCents: state.holdings.reduce((sum, item) => sum + item.amountCents, 0),
    };
  }

  auditLog(sessionId?: string, limit = 20): { records: unknown[]; chain: { ok: boolean; total: number; brokenAt: number | null } } {
    return { records: this.options.chain.list(sessionId, limit), chain: this.options.chain.verify() };
  }
}
