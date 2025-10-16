import json
import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class TokenCounterCallback(BaseCallbackHandler):
    """Callback для подсчета токенов."""

    def __init__(self):
        self.total_tokens = None
        self.completion_tokens = None
        self.prompt_tokens = None
        self.reset()

    def reset(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.total_tokens = 0

    def on_llm_end(self, response, **kwargs):
        if hasattr(response, 'llm_output') and response.llm_output:
            token_usage = response.llm_output.get('token_usage', {})
            self.prompt_tokens = token_usage.get('prompt_tokens', 0)
            self.completion_tokens = token_usage.get('completion_tokens', 0)
            self.total_tokens = token_usage.get('total_tokens', 0)


class InMemoryChatMessageHistory(BaseChatMessageHistory, BaseModel):
    """Простая реализация хранения истории чата в памяти."""
    messages: List[BaseMessage] = Field(default_factory=list)

    def add_messages(self, messages: List[BaseMessage]) -> None:
        """Добавляет список сообщений в хранилище."""
        self.messages.extend(messages)

    def clear(self) -> None:
        """Очищает все сообщения."""
        self.messages = []


class EcommerceLangChainBot:
    def __init__(self):
        load_dotenv()

        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = os.getenv("OPENAI_MODEL")
        self.brand_name = os.getenv("BRAND_NAME")

        if not self.api_key:
            raise ValueError("OPENAI_API_KEY не найден в .env файле")

        self.logs_dir = Path("logs")
        self.logs_dir.mkdir(exist_ok=True)

        self.session_id = str(uuid.uuid4())
        self.logger = self._setup_logging()

        self.faq_data = self._load_json("data/faq.json")
        self.orders = self._load_json("data/orders.json")

        self.store = {}
        self.token_callback = TokenCounterCallback()

        self.llm = ChatOpenAI(
            api_key=self.api_key,
            model=self.model,
            temperature=0.0,
            max_tokens=200,
            timeout=5,
            callbacks=[self.token_callback]
        )

        self.chain_with_history = self._create_chain_with_history()

        self.logger.info(f"Бот {self.brand_name} инициализирован", extra={
            "faq_count": len(self.faq_data),
            "orders_count": len(self.orders),
            "model": self.model
        })

    def _setup_logging(self):
        """Настройка системы логирования."""
        log_filename = self.logs_dir / f"session_{self.session_id[:8]}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        logger = logging.getLogger(f"bot_session_{self.session_id[:8]}")
        logger.setLevel(logging.INFO)

        logger.handlers.clear()

        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-8s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        file_handler = logging.FileHandler(log_filename, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.WARNING)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        logger.info(f"Сессия {self.session_id} начата", extra={
            "session_id": self.session_id,
            "log_file": str(log_filename)
        })

        return logger

    def _load_json(self, filepath: str) -> Dict[str, Any]:
        """Загрузка JSON файлов."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                self.logger.info(f"Загружен файл {filepath}", extra={
                    "filepath": filepath,
                    "items_count": len(data)
                })
                return data
        except FileNotFoundError:
            self.logger.error(f"Файл {filepath} не найден!", extra={"filepath": filepath})
            return {} if filepath.endswith('orders.json') else []
        except json.JSONDecodeError as e:
            self.logger.error(f"Ошибка декодирования {filepath}: {e}", extra={
                "filepath": filepath,
                "error": str(e)
            })
            return {} if filepath.endswith('orders.json') else []

    def _get_order_info(self, order_id: str) -> str:
        """Получение информации о заказе для передачи в контекст LLM."""
        if order_id not in self.orders:
            self.logger.warning(f"Запрос несуществующего заказа: {order_id}", extra={
                "order_id": order_id,
                "available_orders": list(self.orders.keys())
            })
            return f"ЗАКАЗ НЕ НАЙДЕН: заказ #{order_id} отсутствует в системе."

        order_data = self.orders[order_id]
        status = order_data.get("status", "unknown")

        self.logger.info(f"Запрос информации о заказе {order_id}", extra={
            "order_id": order_id,
            "status": status,
            "order_data": order_data
        })

        order_info = f"ИНФОРМАЦИЯ О ЗАКАЗЕ #{order_id}:\n"
        order_info += f"- Статус: {status}\n"

        if status == "processing":
            order_info += f"- Примечание: {order_data.get('note', 'Обрабатывается')}\n"
        elif status == "in_transit":
            order_info += f"- Ожидаемая доставка: {order_data.get('eta_days', 'не указано')} дней\n"
            order_info += f"- Перевозчик: {order_data.get('carrier', 'не указан')}\n"
        elif status == "delivered":
            order_info += f"- Дата доставки: {order_data.get('delivered_at', 'не указана')}\n"
        elif status == "cancelled":
            order_info += f"- Причина отмены: {order_data.get('note', 'По запросу клиента')}\n"

        return order_info

    def _create_faq_context(self) -> str:
        """Создание контекста FAQ для промпта."""
        if not self.faq_data:
            return "FAQ база недоступна."

        faq_text = "БАЗА ЗНАНИЙ FAQ:\n"
        for i, item in enumerate(self.faq_data, 1):
            faq_text += f"{i}. Вопрос: {item['q']}\n   Ответ: {item['a']}\n\n"

        return faq_text.strip()

    def _create_orders_context(self) -> str:
        """Создание контекста заказов для промпта."""
        if not self.orders:
            return "База заказов недоступна."

        orders_text = "ДОСТУПНЫЕ ЗАКАЗЫ В СИСТЕМЕ:\n"
        for order_id, data in self.orders.items():
            orders_text += f"- Заказ #{order_id}: статус '{data.get('status', 'unknown')}'\n"

        return orders_text

    def _create_chain_with_history(self):
        """Создание цепочки LangChain с поддержкой истории."""

        faq_context = self._create_faq_context()
        orders_context = self._create_orders_context()

        system_prompt = f"""Ты бот поддержки интернет-магазина {self.brand_name}.

{faq_context}

{orders_context}

ИНСТРУКЦИИ:
- Отвечай кратко, вежливо и по делу (максимум 2-3 предложения)
- ОБЯЗАТЕЛЬНО используй информацию из FAQ выше для ответов на типовые вопросы
- При запросах о заказах используй информацию, которую получишь в контексте сообщения
- Если информации нет в FAQ или о заказе - честно скажи и предложи обратиться в поддержку
- НЕ выдумывай информацию, которой нет в базе знаний
- Общайся на русском языке
- Будь дружелюбным и профессиональным

ОБРАБОТКА КОМАНД /order:
- Когда пользователь пишет "/order номер", анализируй предоставленную информацию о заказе
- Представь статус заказа в удобном для пользователя виде
- Если заказ не найден - предложи проверить номер или обратиться в поддержку

ВАЖНО: Ты видишь всю базу FAQ и список заказов выше - используй эту информацию!"""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])

        chain = prompt | self.llm | StrOutputParser()

        chain_with_history = RunnableWithMessageHistory(
            chain,
            self._get_session_history,
            input_messages_key="input",
            history_messages_key="history",
        )

        return chain_with_history

    def _get_session_history(self, session_id: str) -> BaseChatMessageHistory:
        """Получение истории сессии."""
        if session_id not in self.store:
            self.store[session_id] = InMemoryChatMessageHistory()
        return self.store[session_id]

    def _log_interaction(self, user_input: str, bot_response: str, usage: Dict[str, int]):
        """Логирование взаимодействия."""
        self.logger.info("Взаимодействие", extra={
            "user_input": user_input,
            "bot_response": bot_response,
            "token_usage": usage,
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
            "total_tokens": usage.get("total_tokens", 0)
        })

    def _process_user_input(self, user_input: str) -> tuple[str, Dict[str, int]]:
        """Обработка пользовательского ввода."""
        self.token_callback.reset()

        enhanced_input = user_input
        if user_input.strip().startswith('/order'):
            parts = user_input.strip().split()
            if len(parts) == 2:
                order_id = parts[1]
                order_info = self._get_order_info(order_id)
                enhanced_input = f"{user_input}\n\n{order_info}"
                self.logger.info(f"Обработка команды /order для заказа {order_id}")
            else:
                enhanced_input = f"{user_input}\n\nОШИБКА: Неверный формат команды. Используйте: /order <номер_заказа>"
                self.logger.warning("Неверный формат команды /order", extra={"input": user_input})

        config = {"configurable": {"session_id": self.session_id}}

        try:
            response = self.chain_with_history.invoke(
                {"input": enhanced_input},
                config=config
            )

            usage = {
                "prompt_tokens": self.token_callback.prompt_tokens,
                "completion_tokens": self.token_callback.completion_tokens,
                "total_tokens": self.token_callback.total_tokens
            }

            self.logger.info("Успешная обработка через LLM", extra={
                "input_length": len(user_input),
                "response_length": len(response),
                "usage": usage
            })

            return response, usage

        except Exception as e:
            error_msg = f"Извините, произошла ошибка. Попробуйте позже или обратитесь в службу поддержки."
            self.logger.error(f"Ошибка LangChain: {e}", extra={
                "error": str(e),
                "user_input": user_input
            })
            return error_msg, {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def run(self):
        """Основной цикл бота."""
        self.logger.info("Запуск основного цикла бота")

        print(f"🛍️ Добро пожаловать в поддержку {self.brand_name}!")
        print("Я помогу с вопросами о заказах, доставке и возврате.")
        print("Команды: /order <номер> - статус заказа, /help - справка, /exit - выход")
        print(f"📚 FAQ база загружена в контекст ({len(self.faq_data)} записей)")
        print(f"📦 Заказов в системе: {len(self.orders)}")
        print(f"📋 Логи сохраняются в: logs/session_{self.session_id[:8]}_*.jsonl\n")

        interaction_count = 0

        while True:
            try:
                user_input = input("Вы: ").strip()

                if not user_input:
                    continue

                interaction_count += 1

                if user_input.lower() in ['/exit', '/quit', 'выход']:
                    self.logger.info(f"Завершение сессии. Всего взаимодействий: {interaction_count}")
                    break

                if user_input.lower() == '/help':
                    help_text = f"""
📋 Доступные команды:
• /order <номер> - проверить статус заказа
• /help - показать эту справку  
• /exit - завершить сессию

🤖 Возможности:
• Полная база FAQ загружена в контекст LLM
• ВСЕ запросы проходят через LLM для естественных ответов
• Поддержка истории диалога через LangChain
• Интеллектуальные ответы на вопросы с опечатками

❓ Примеры вопросов:
• Как оформить возврат?
• Сколько идёт доставка?
• Какие способы оплаты принимаете?
• Как применить промокод?

📊 Доступные заказы для теста: {', '.join(self.orders.keys())}
                    """
                    print(f"Бот: {help_text}")
                    continue

                bot_response, usage = self._process_user_input(user_input)

                self._log_interaction(user_input, bot_response, usage)

                print(f"Бот: {bot_response}")

                if usage["total_tokens"] > 0:
                    self.logger.info(
                        f"💰 Токены: {usage['total_tokens']} (prompt: {usage['prompt_tokens']}, completion: {usage['completion_tokens']})")


            except KeyboardInterrupt:
                self.logger.info(f"Сессия прервана пользователем. Взаимодействий: {interaction_count}")
                break
            except Exception as e:
                self.logger.error(f"Неожиданная ошибка в основном цикле: {e}")


def main():
    """Точка входа в приложение."""
    bot = EcommerceLangChainBot()
    bot.run()


if __name__ == "__main__":
    main()
