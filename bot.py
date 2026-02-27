import asyncio
import datetime as dt
import html
import json
import logging
import math
import os
from collections import Counter
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import duckdb
from openai import OpenAI
from telegram import ReplyKeyboardMarkup, ReplyKeyboardRemove, Update
from telegram.ext import (
    Application,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

TOPIC_BUTTONS = {
    "Задача на group by": "group_agg",
    "Задача на join": "join",
    "Задача на where": "filter",
}
NEXT_TASK_BUTTON = "Следующая задача"
TOPIC_KEYBOARD = ReplyKeyboardMarkup(
    [["Задача на group by", "Задача на join", "Задача на where"]],
    resize_keyboard=True,
)
NEXT_TASK_KEYBOARD = ReplyKeyboardMarkup([[NEXT_TASK_BUTTON]], resize_keyboard=True)

TASK_SYSTEM_PROMPT = """Ты генератор SQL-задач для тренажера.
Ответь строго валидным JSON без markdown-блоков и без пояснений.
Тема должна соответствовать входному topic:
- group_agg -> задача на GROUP BY/агрегации
- join -> задача на JOIN
- filter -> задача на WHERE/фильтрацию
Верни объект ровно в формате:
{
  "topic": "group_agg | join | filter",
  "title": "string",
  "task": {
    "problem": "string",
    "expected_output": {
      "columns": [
        { "name": "string", "type": "string", "description": "string" }
      ],
      "ordering_required": false
    }
  },
  "schema": {
    "tables": [
      {
        "name": "string",
        "ddl": "string"
      }
    ]
  },
  "data": {
    "inline_rows": {
      "table_name": [
        { "col": "value" }
      ]
    }
  },
  "reference": {
    "sql": "string"
  }
}
Правила:
- Язык задания: русский.
- Данные компактные, но достаточные для проверки.
- Каждая генерация должна быть заметно отличной от типовых учебных примеров.
- Избегай шаблонов вроде employees/departments, «сотрудники и отделы», «заказы и клиенты», если можно придумать альтернативу.
- По возможности используй разные сеттинги: логистика, финансы, подписки/биллинг, медицина, спорт, образование, соцсети, производство, телеметрия, маркетинг, саппорт, каталог товаров, путешествия.
- Иногда добавляй реалистичные нюансы данных (в пределах компактности): NULL, дубликаты, несколько строк на сущность, «пустые» категории, значения на границе (0, отрицательные, одинаковые суммы), разные даты.
- Для topic=group_agg или topic=filter генерируй одну таблицу длиной от 5 до 12 строк.
- Для topic=join каждая таблица должна содержать не менее 2 и не более 6 строк.
- DDL должен быть исполнимым в DuckDB.
- reference.sql должен корректно решать задачу.
- expected_output.columns должно соответствовать reference.sql по количеству и смыслу.
- Не используй таблицы с более чем 3 столбцами. Название столбцов выбирай покороче, не более 10 символов
"""

FEEDBACK_SYSTEM_PROMPT = """Ты SQL-ревьюер в Telegram-тренажере.
Тебе передают факт-вердикт от бэкенда. Не меняй status.
Ответь строго JSON без markdown и пояснений в формате:
{
  "status": "execution_error | wrong_answer | correct",
  "summary": "string",
  "issues": [
    {
      "type": "execution | logic | quality",
      "severity": "error | warning | info",
      "message": "string",
      "hint": "string"
    }
  ]
}
Требования:
- summary: 1-2 предложения, коротко для Telegram.
- execution_error: расшифруй ошибку понятным языком и дай 1-3 конкретных правки.
- wrong_answer: объясни расхождение на основе diff, без полного решения.
- correct: можно 1-2 улучшения по качеству/стилю.
- issues можно опустить только если действительно не нужно.
"""


@dataclass
class TaskState:
    raw_task: dict[str, Any]
    expected_rows: list[tuple[Any, ...]]
    expected_columns_count: int
    ordering_required: bool
    schema_ddl: list[str]
    solved: bool = False


def get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Не найден OPENAI_API_KEY в переменных окружения")
    return OpenAI(api_key=api_key)


def get_openai_model() -> str:
    model = os.getenv("OPENAI_MODEL")
    if not model:
        raise RuntimeError("Не найден OPENAI_MODEL в переменных окружения")
    return model


def extract_json(raw_text: str) -> dict[str, Any]:
    text = raw_text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        if len(parts) >= 2:
            text = parts[1]
            if text.startswith("json"):
                text = text[4:]
    return json.loads(text.strip())


def call_openai_json(
    system_prompt: str,
    user_payload: dict[str, Any],
    temperature: float = 0.2,
) -> dict[str, Any]:
    client = get_openai_client()
    model = get_openai_model()
    completion = client.chat.completions.create(
        model=model,
        temperature=temperature,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
        ],
    )
    content = completion.choices[0].message.content or "{}"
    return extract_json(content)


def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def build_db_and_seed(task_json: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    conn = duckdb.connect(database=":memory:")
    tables = task_json["schema"]["tables"]
    inline_rows = task_json["data"]["inline_rows"]

    for t in tables:
        conn.execute(t["ddl"])

    for table_name, rows in inline_rows.items():
        if not rows:
            continue
        cols = list(rows[0].keys())
        placeholders = ", ".join(["?"] * len(cols))
        quoted_cols = ", ".join(quote_ident(c) for c in cols)
        sql = f"INSERT INTO {quote_ident(table_name)} ({quoted_cols}) VALUES ({placeholders})"
        values = [tuple(r.get(c) for c in cols) for r in rows]
        conn.executemany(sql, values)

    return conn


def normalize_scalar(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    return value


def normalize_rows(rows: list[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return [tuple(normalize_scalar(v) for v in row) for row in rows]


def tuple_rows_to_preview(rows: list[tuple[Any, ...]], limit: int = 5) -> list[list[Any]]:
    return [list(r) for r in rows[:limit]]


def multiset_diff(
    expected: list[tuple[Any, ...]],
    actual: list[tuple[Any, ...]],
) -> tuple[list[tuple[Any, ...]], list[tuple[Any, ...]]]:
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    missing = list((expected_counter - actual_counter).elements())
    unexpected = list((actual_counter - expected_counter).elements())
    return missing, unexpected


def evaluate_user_sql(task_state: TaskState, user_sql: str) -> dict[str, Any]:
    conn = build_db_and_seed(task_state.raw_task)
    try:
        cursor = conn.execute(user_sql)
        result = cursor.fetchall()
        actual_columns_count = len(cursor.description or [])
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "execution_error",
            "execution_error": {"message": str(exc)},
        }
    finally:
        conn.close()

    actual_rows = normalize_rows(result)
    expected_rows = task_state.expected_rows

    if actual_columns_count != task_state.expected_columns_count:
        missing, unexpected = multiset_diff(expected_rows, actual_rows)
        return {
            "status": "wrong_answer",
            "diff": {
                "expected_row_count": len(expected_rows),
                "actual_row_count": len(actual_rows),
                "expected_preview": tuple_rows_to_preview(expected_rows),
                "actual_preview": tuple_rows_to_preview(actual_rows),
                "missing_rows_preview": tuple_rows_to_preview(missing),
                "unexpected_rows_preview": tuple_rows_to_preview(unexpected),
            },
        }

    if task_state.ordering_required:
        is_equal = expected_rows == actual_rows
    else:
        is_equal = Counter(expected_rows) == Counter(actual_rows)

    if is_equal:
        return {"status": "correct"}

    missing, unexpected = multiset_diff(expected_rows, actual_rows)
    return {
        "status": "wrong_answer",
        "diff": {
            "expected_row_count": len(expected_rows),
            "actual_row_count": len(actual_rows),
            "expected_preview": tuple_rows_to_preview(expected_rows),
            "actual_preview": tuple_rows_to_preview(actual_rows),
            "missing_rows_preview": tuple_rows_to_preview(missing),
            "unexpected_rows_preview": tuple_rows_to_preview(unexpected),
        },
    }


def validate_task_json(task_json: dict[str, Any]) -> None:
    required_paths = [
        ("topic",),
        ("title",),
        ("task", "problem"),
        ("task", "expected_output", "columns"),
        ("task", "expected_output", "ordering_required"),
        ("schema", "tables"),
        ("data", "inline_rows"),
        ("reference", "sql"),
    ]
    for path in required_paths:
        cur: Any = task_json
        for key in path:
            if key not in cur:
                joined = ".".join(path)
                raise ValueError(f"В ответе модели отсутствует поле `{joined}`")
            cur = cur[key]


def render_inline_table(table_name: str, rows: list[dict[str, Any]], limit: int = 10) -> str:
    if not rows:
        return f"{table_name}\n(пусто)"

    cols = list(rows[0].keys())
    shown_rows = rows[:limit]
    str_rows = [[str(r.get(c)) for c in cols] for r in shown_rows]
    widths = [len(c) for c in cols]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    header = " | ".join(cols[i].ljust(widths[i]) for i in range(len(cols)))
    sep = "-+-".join("-" * widths[i] for i in range(len(cols)))
    body = "\n".join(
        " | ".join(row[i].ljust(widths[i]) for i in range(len(cols))) for row in str_rows
    )
    trunc = ""
    if len(rows) > limit:
        trunc = f"\n... и еще {len(rows) - limit} строк"
    return f"{table_name}\n{header}\n{sep}\n{body}{trunc}"


def task_to_message(task_json: dict[str, Any]) -> str:
    title = task_json["title"]
    problem = task_json["task"]["problem"]
    columns = task_json["task"]["expected_output"]["columns"]
    inline_rows = task_json["data"]["inline_rows"]

    cols_text = "\n".join([f"- {c['name']} ({c['type']})" for c in columns])
    tables_rendered = []
    for table_name, rows in inline_rows.items():
        tables_rendered.append(render_inline_table(table_name, rows))

    tables_block = "\n\n".join(tables_rendered)
    escaped_tables = html.escape(tables_block)
    return (
        f"<b>{html.escape(title)}</b>\n\n"
        f"{html.escape(problem)}\n\n"
        f"<b>Ожидаемые колонки:</b>\n{html.escape(cols_text)}\n\n"
        f"<b>Таблицы и данные:</b>\n<pre>{escaped_tables}</pre>\n\n"
        "Отправь свой SQL-запрос одним сообщением."
    )


def build_feedback_input(task_state: TaskState, user_sql: str, verdict: dict[str, Any]) -> dict[str, Any]:
    expected_output = task_state.raw_task["task"]["expected_output"]
    payload: dict[str, Any] = {
        "task": {
            "problem": task_state.raw_task["task"]["problem"],
            "expected_output": {
                "columns": [
                    {"name": c["name"], "type": c["type"]}
                    for c in expected_output["columns"]
                ],
                "ordering_required": expected_output["ordering_required"],
            },
        },
        "schema_ddl": task_state.schema_ddl,
        "user_sql": user_sql,
        "backend_verdict": {"status": verdict["status"]},
    }
    if verdict["status"] == "execution_error":
        payload["backend_verdict"]["execution_error"] = verdict["execution_error"]
    if verdict["status"] == "wrong_answer":
        payload["backend_verdict"]["diff"] = verdict["diff"]
    return payload


def fallback_feedback(verdict: dict[str, Any]) -> dict[str, Any]:
    status = verdict["status"]
    if status == "execution_error":
        return {
            "status": "execution_error",
            "summary": "Запрос не выполнился. Проверь синтаксис и имена таблиц/колонок.",
            "issues": [
                {
                    "type": "execution",
                    "severity": "error",
                    "message": verdict["execution_error"]["message"],
                    "hint": "Проверь SELECT/FROM/JOIN и запятые между колонками.",
                }
            ],
        }
    if status == "wrong_answer":
        diff = verdict["diff"]
        return {
            "status": "wrong_answer",
            "summary": "Результат запроса отличается от ожидаемого.",
            "issues": [
                {
                    "type": "logic",
                    "severity": "warning",
                    "message": (
                        f"Ожидалось строк: {diff['expected_row_count']}, "
                        f"получено: {diff['actual_row_count']}."
                    ),
                    "hint": "Проверь условия JOIN/WHERE и поля группировки.",
                }
            ],
        }
    return {
        "status": "correct",
        "summary": "Верно! Результат совпадает с эталоном.",
        "issues": [
            {
                "type": "quality",
                "severity": "info",
                "message": "Решение корректно.",
                "hint": "Можно улучшать читаемость через алиасы и явные имена колонок.",
            }
        ],
    }


def generate_feedback(task_state: TaskState, user_sql: str, verdict: dict[str, Any]) -> dict[str, Any]:
    payload = build_feedback_input(task_state, user_sql, verdict)
    try:
        feedback = call_openai_json(FEEDBACK_SYSTEM_PROMPT, payload)
        if "status" not in feedback or "summary" not in feedback:
            raise ValueError("feedback missing required keys")
        feedback["status"] = verdict["status"]
        return feedback
    except Exception:  # noqa: BLE001
        logger.exception("Не удалось получить sql_feedback от OpenAI, используем fallback")
        return fallback_feedback(verdict)


def generate_task(topic: str) -> TaskState:
    task_json = call_openai_json(TASK_SYSTEM_PROMPT, {"topic": topic}, temperature=0.8)
    validate_task_json(task_json)
    if task_json["topic"] != topic:
        raise ValueError("Тема задачи не совпала с выбранной кнопкой")

    conn = build_db_and_seed(task_json)
    try:
        expected_rows = conn.execute(task_json["reference"]["sql"]).fetchall()
    finally:
        conn.close()

    expected_rows_norm = normalize_rows(expected_rows)
    columns = task_json["task"]["expected_output"]["columns"]
    schema_ddl = [t["ddl"] for t in task_json["schema"]["tables"]]
    return TaskState(
        raw_task=task_json,
        expected_rows=expected_rows_norm,
        expected_columns_count=len(columns),
        ordering_required=task_json["task"]["expected_output"]["ordering_required"],
        schema_ddl=schema_ddl,
    )


def get_task_state(context: ContextTypes.DEFAULT_TYPE) -> TaskState | None:
    state = context.user_data.get("task_state")
    if not state:
        return None
    return state


def reset_flow(context: ContextTypes.DEFAULT_TYPE) -> None:
    context.user_data.pop("task_state", None)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    reset_flow(context)
    await update.message.reply_text(
        "Выбери тему задачи:",
        reply_markup=TOPIC_KEYBOARD,
    )


async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = (update.message.text or "").strip()
    if not text:
        return

    if text == NEXT_TASK_BUTTON:
        reset_flow(context)
        await update.message.reply_text(
            "Начинаем новую задачу. Выбери тему:",
            reply_markup=TOPIC_KEYBOARD,
        )
        return

    if text in TOPIC_BUTTONS:
        topic = TOPIC_BUTTONS[text]
        await update.message.reply_text("Генерирую задачу, это может занять несколько секунд...")
        try:
            task_state = await asyncio.to_thread(generate_task, topic)
        except Exception as exc:  # noqa: BLE001
            logger.exception("Ошибка при генерации задачи")
            await update.message.reply_text(f"Не удалось создать задачу: {exc}")
            return

        context.user_data["task_state"] = task_state
        task_text = task_to_message(task_state.raw_task)
        await update.message.reply_text(task_text, parse_mode="HTML", reply_markup=ReplyKeyboardRemove())
        return

    task_state = get_task_state(context)
    if not task_state:
        await update.message.reply_text(
            "Сначала выбери тему задачи кнопкой ниже.",
            reply_markup=TOPIC_KEYBOARD,
        )
        return

    if task_state.solved:
        await update.message.reply_text(
            "Эта задача уже решена. Нажми «Следующая задача».",
            reply_markup=NEXT_TASK_KEYBOARD,
        )
        return

    user_sql = text
    verdict = await asyncio.to_thread(evaluate_user_sql, task_state, user_sql)
    feedback = await asyncio.to_thread(generate_feedback, task_state, user_sql, verdict)
    summary = feedback.get("summary", "Проверка завершена.")

    await update.message.reply_text(summary)
    if verdict["status"] == "correct":
        task_state.solved = True
        await update.message.reply_text(
            "Задача решена. Нажми «Следующая задача», чтобы продолжить.",
            reply_markup=NEXT_TASK_KEYBOARD,
        )


def main() -> None:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    if not token:
        raise RuntimeError("Не найден TELEGRAM_BOT_TOKEN в переменных окружения")

    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))

    logger.info("Bot started")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()
