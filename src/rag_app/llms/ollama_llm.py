from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from .base import BaseLLM


class OllamaChatModel(BaseLLM):
    """Chat model served by a local Ollama server (uses Ollama's /api/chat).

    The prompt string is sent as one user message, optionally preceded by a
    system message, and only the answer text is returned, so the BaseLLM
    contract (str in, str out) stays the same.

    Args:
        model_name: Ollama model tag (must be pulled first).
        base_url: Ollama server URL.
        temperature: 0 = deterministic, higher = more creative.
        num_ctx: Context window in tokens; prompts longer than this are cut silently.
        reasoning: None = don't send the option (use for instruct models).
            True = thinking models only; the thinking is kept out of the answer.
            Instruct models reject True.
        system_prompt: Optional standing instructions sent as a system message
            before every prompt (e.g. "Answer only from the context.").
    Raises:
        ResponseError: If the model is not pulled (checked at creation).
        ConnectionError: If the Ollama server is not reachable.
    """

    name = "ollama"

    def __init__(
        self,
        model_name: str = "qwen3:4b-instruct",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.0,
        num_ctx: int = 8192,
        reasoning: bool | None = None,
        system_prompt: str | None = None,
    ):
        self.model_name = model_name
        self.system_prompt = system_prompt

        self.chat = ChatOllama(
            model=model_name,
            base_url=base_url,
            temperature=temperature,
            num_ctx=num_ctx,
            reasoning=reasoning,
            validate_model_on_init=True,
        )

    def _messages(self, prompt: str) -> list[BaseMessage]:
        """Turn a plain prompt string into the message list a chat model expects.

        A chat model does not take a single string; it takes a list of
        messages, each labelled with who "said" it:
            SystemMessage -> standing rules for the model (how to behave)
            HumanMessage  -> what the user is asking right now

        This helper builds that list so generate() and stream() can keep
        accepting a plain string (the BaseLLM contract).

        Args:
            prompt: The full text to send as the user's message
                (in RAG: the context chunks + the question).
        Returns:
            Without a system_prompt:
                [HumanMessage(prompt)]
            With a system_prompt:
                [SystemMessage(system_prompt), HumanMessage(prompt)]

        Example:
            >>> llm = OllamaChatModel(system_prompt="Be brief.")
            >>> llm._messages("What is RAG?")
            [SystemMessage(content='Be brief.'), HumanMessage(content='What is RAG?')]
        """
        messages: list[BaseMessage] = []
        if self.system_prompt:
            messages.append(SystemMessage(content=self.system_prompt))
        messages.append(HumanMessage(content=prompt))
        return messages

    def generate(self, prompt):
        """Ask the model and wait for the complete answer.

        Steps:
            1. _messages(prompt) wraps the string as [system?, user] messages.
            2. self.chat.invoke(...) sends them to Ollama and returns an
               AIMessage (the model's reply, plus metadata like token counts).
            3. .content takes just the answer text out of that AIMessage.

        Args:
            prompt: The full text to send (context + question in RAG).
        Returns:
            The answer as a plain string.
        """
        return self.chat.invoke(self._messages(prompt)).content

    def stream(self, prompt):
        """Ask the model and get the answer piece by piece, as it is written.

        Use this for live output (text appears word by word, like ChatGPT)
        instead of waiting for the whole answer.

        Steps:
            1. _messages(prompt) wraps the string as [system?, user] messages.
            2. self.chat.stream(...) yields small AIMessageChunk objects.
            3. We yield each chunk's .content (the text piece).
            4. Empty pieces are skipped: Ollama sometimes sends chunks with no
               answer text (e.g. while a thinking model reasons, or the final
               metadata chunk).

        Args:
            prompt: The full text to send (context + question in RAG).
        Yields:
            Text pieces; "".join(all pieces) == generate(prompt).

        Example:
            >>> for piece in llm.stream("What is RAG?"):
            ...     print(piece, end="", flush=True)
        """
        for chunk in self.chat.stream(self._messages(prompt)):
            if chunk.content:
                yield chunk.content
