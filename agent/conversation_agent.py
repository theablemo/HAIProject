from typing import List, Optional, Dict, Any
from datetime import datetime
import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import AzureChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
import json
from agent.models import Conversation, Solution, Insight

class ConversationAgent:
    def __init__(
        self,
        model_name: str = "gemini-2.0-flash",
        temperature: float = 0.1,
        max_tokens: int = 1000,
        memory_key: str = "chat_history",
    ):
        """
        Initialize the ConversationAgent with Azure OpenAI model and conversation memory.

        Args:
            model_name (str): Name of the model to use
            temperature (float): Temperature for model sampling
            max_tokens (int): Maximum number of tokens to generate
            memory_key (str): Key for storing conversation history in memory
        """
        # Load environment variables
        load_dotenv()

        # Initialize Azure OpenAI chat model
        self.llm = AzureChatOpenAI(
            deployment_name="gpt-4o",
            model_name="gpt-4o",
            temperature=temperature,
            # max_tokens=max_tokens,
            azure_endpoint=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_ENDPOINT"),
            api_key=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_KEY"),
            api_version=os.getenv("AZURE_OPENAI_CHAT_COMPLETION_API_VERSION"),
        )

        # Commented out Google's Gemini implementation
        # self.llm = ChatGoogleGenerativeAI(
        #     model=model_name,
        #     temperature=temperature,
        #     max_output_tokens=max_tokens,
        #     google_api_key=os.getenv("GOOGLE_API_KEY"),
        # )

        # Initialize chat prompt template
        self.prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="You are a helpful AI assistant."),
            MessagesPlaceholder(variable_name=memory_key),
            HumanMessage(content="{input}")
        ])

        # Initialize the chain using the new pattern
        self.chain = (
            {"input": RunnablePassthrough(), "chat_history": RunnablePassthrough()}
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        # Initialize conversation history
        self.chat_history = []

        # Initialize conversation object
        self.conversation = Conversation(
            problem_text="",  # Will be set when starting a conversation
            solutions=[],
            insights=[],
            background_info={},
            chat_history=[]
        )

    def start_conversation(self, problem_text: str):
        """
        Start a new conversation with a problem description.

        Args:
            problem_text (str): Description of the problem or topic
        """
        self.conversation = Conversation(
            problem_text=problem_text,
            solutions=[],
            insights=[],
            background_info={},
            chat_history=[]
        )
        self.chat_history = []

    def process_message(self, message: str) -> str:
        """
        Process a user message and return the AI's response.

        Args:
            message (str): The user's message to process

        Returns:
            str: The AI's response
        """
        # Add user message to both memory and conversation
        self.conversation.add_message("user", message)
        
        # Process message with the chain
        response = self.chain.invoke({
            "input": message,
            "chat_history": self.chat_history
        })
        
        # Add AI response to both memory and conversation
        self.conversation.add_message("assistant", response)
        self.chat_history.append(HumanMessage(content=message))
        self.chat_history.append(AIMessage(content=response))
        
        return response

    def add_solution(self, title: str, subtitle: str):
        """
        Add a solution to the conversation.

        Args:
            title (str): Title of the solution
            subtitle (str): Subtitle/description of the solution
        """
        solution = Solution(title=title, subtitle=subtitle)
        self.conversation.add_solution(solution)

    def add_insight(self, text: str, sources: List[str], vega_lite_spec: Optional[Dict[str, Any]] = None):
        """
        Add an insight to the conversation.

        Args:
            text (str): Text of the insight
            sources (List[str]): List of source references
            vega_lite_spec (Optional[Dict[str, Any]]): Optional visualization specification
        """
        insight = Insight(text=text, sources=sources, vega_lite_spec=vega_lite_spec)
        self.conversation.add_insight(insight)

    def end_conversation(self) -> Dict[str, Any]:
        """
        End the current conversation and save it to the database.

        Returns:
            Dict[str, Any]: The saved conversation data
        """
        # Prepare conversation data for storage
        conversation_data = {
            "id": str(datetime.now().timestamp()),
            "timestamp": datetime.now().isoformat(),
            "problem_text": self.conversation.problem_text,
            "solutions": [
                {"title": s.title, "subtitle": s.subtitle}
                for s in self.conversation.solutions
            ],
            "insights": [
                {
                    "text": i.text,
                    "sources": i.sources,
                    "vega_lite_spec": i.vega_lite_spec
                }
                for i in self.conversation.insights
            ],
            "chat_history": self.conversation.chat_history
        }

        # Save to database (implement your database logic here)
        self._save_to_database(conversation_data)

        return conversation_data

    def get_conversations(self) -> List[Dict[str, Any]]:
        """
        Retrieve all conversations from the database.

        Returns:
            List[Dict[str, Any]]: List of all conversations
        """
        return self._load_from_database()

    def _save_to_database(self, conversation_data: Dict[str, Any]):
        """
        Save conversation data to the database.
        Implement your database logic here.

        Args:
            conversation_data (Dict[str, Any]): The conversation data to save
        """
        # TODO: Implement database storage
        # For now, we'll just save to a JSON file
        conversations = self._load_from_database()
        conversations.append(conversation_data)
        
        with open("conversations.json", "w") as f:
            json.dump(conversations, f, indent=2)

    def _load_from_database(self) -> List[Dict[str, Any]]:
        """
        Load all conversations from the database.

        Returns:
            List[Dict[str, Any]]: List of all conversations
        """
        # TODO: Implement database retrieval
        # For now, we'll just load from a JSON file
        try:
            with open("conversations.json", "r") as f:
                return json.load(f)
        except FileNotFoundError:
            return []
