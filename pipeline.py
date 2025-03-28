from typing import List, Dict, Any, Tuple
from conversation import Conversation
from conversation_agent import ConversationAgent
from langchain.chains import LLMChain

from AzureSTT import AzureSpeechToText
from embedding_manager import EmbeddingManager

from langchain.agents import AgentExecutor
from langchain.prompts import ChatPromptTemplate
from langchain.agents.agent_types import AgentType
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import json
import io

class Pipeline:
    def __init__(self):
        """Initialize the pipeline with all necessary components."""
        # Initialize components
        self.conversation_agent = ConversationAgent()
        self.speech_to_text = AzureSpeechToText()
        self.embedding_manager = EmbeddingManager()
        
        # Initialize agents for different tasks
        self.conversation_analysis_chain = self._create_conversation_analysis_chain()
        self.text_insight_chain = self._create_text_insight_chain()
        
    def _create_conversation_analysis_chain(self) -> LLMChain:
        """Create a chain for analyzing and updating problem description, background information, and solutions."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant analyzing a continuous medical dialogue between a doctor and a patient.
            Your task is to analyze the transcribed audio segment and update three aspects of the conversation:
            1. Problem description
            2. Background information
            3. Solutions
            
            Important context:
            1. This is part of an ongoing medical consultation
            2. You have access to:
               - The current problem description (if any exists)
               - The current background information collected so far (if any exists)
               - The current solutions (if any exist)
            3. You should analyze each aspect independently:
               - Problem: Update if new medical condition information is present
               - Background: Update with any new patient information, medical history, or relevant context
               - Solutions: Identify any new treatment options or solutions mentioned
            
            You should:
            1. Analyze the new transcription for each aspect:
               a. Problem description:
                  - If no current problem exists and a medical problem is mentioned: Create new problem description
                  - If current problem exists: Update only if new information about the problem is present
               
               b. Background information:
                  - Compare with current background info
                  - Add any new patient information, medical history, symptoms, or relevant context
                  - Update existing background info if new details are provided
               
               c. Solutions:
                  - Compare with current solutions
                  - Add any new treatment options, medications, or solutions mentioned
                  - Do not duplicate existing solutions
                  - Do not add solutions that are not mentioned in the transcription
                  - Do not add solutions based on your own knowledge, only based on what is mentioned in the transcription
            
            Return a JSON with the following structure:
            {
                "problem_text": "The problem description (updated or new)",
                "background_info": [{"key": "value"}, ...], # Only include new or updated background information
                "solutions": [{"title": "Solution title", "subtitle": "Solution description"}, ...],  # Only include new solutions
                "is_update": true/false  # Indicates if any aspect was updated
            }
            """),
            HumanMessage(content="""Current problem description: {current_problem}
            Current background information: {current_background}
            Current solutions: {current_solutions}
            New transcription: {transcribed_text}
            """)
        ])
        return LLMChain(llm=self.conversation_agent.llm, prompt=prompt)
    
    def _create_text_insight_chain(self) -> LLMChain:
        """Create a chain for generating insights from text documents."""
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant analyzing a continuous medical dialogue between a doctor and a patient.
            Your task is to analyze the provided text documents and generate new insights that are relevant to the current medical consultation.
            
            Important context:
            1. This is part of an ongoing medical consultation
            2. You have access to:
               - The current problem description
               - The current background information
               - The current solutions
               - Previously generated insights
            3. You should:
               - Generate new insights that are strictly based on the provided documents
               - Ensure new insights are not similar to previously generated ones
               - For each insight, specify which documents were used as sources
               - If an insight would be better conveyed through a visualization, generate a Vega-Lite specification
               - For visualization insights, provide a brief explanation of what the chart shows
            
            Return a JSON with the following structure:
            {
                "insights": [
                    {
                        "text": "Insight text or chart explanation",
                        "sources": ["document1", "document2"],
                        "needs_visualization": true/false,
                        "vega_lite_spec": {
                            // Vega-Lite specification if needs_visualization is true
                            // null if needs_visualization is false
                        }
                    }
                ]
            }
            """),
            HumanMessage(content="""Current problem description: {current_problem}
            Current background information: {current_background}
            Current solutions: {current_solutions}
            Previous insights: {previous_insights}
            Documents: {documents}
            """)
        ])
        return LLMChain(llm=self.conversation_agent.llm, prompt=prompt)
    
    def _create_csv_insight_agent(self, csv_docs: List[Tuple[Document, float]]) -> AgentExecutor:
        """Create an agent for generating insights from CSV documents."""
        # Create a DataFrame from all CSV documents
        dfs = []
        for doc, _ in csv_docs:
            df = pd.read_csv(io.StringIO(doc.page_content))
            dfs.append(df)
        
        # Create a pandas DataFrame agent
        agent = create_pandas_dataframe_agent(
            llm=self.conversation_agent.llm,
            dfs=dfs,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        
        return agent
    
    # def process_audio(self, audio_path: str) -> Dict[str, Any]:
    def process_audio(self, transcribed_text: str) -> Conversation:
        """Process an audio file and update the conversation."""
        # Step 1: Transcribe audio
        # transcribed_text = self.speech_to_text.transcribe(audio_path)
        
        # Step 2: Analyze conversation state and update
        analysis = self.conversation_analysis_chain.invoke(
            transcribed_text=transcribed_text,
            current_problem=self.conversation_agent.conversation.problem_text or "No current problem description.",
            current_background=json.dumps(self.conversation_agent.conversation.background_info) or "No current background information.",
            current_solutions=json.dumps([
                {"title": s.title, "subtitle": s.subtitle}
                for s in self.conversation_agent.conversation.solutions
            ]) or "No current solutions."
        )
        analysis_data = json.loads(analysis)
        
        # Update conversation with analysis only if there are updates
        if analysis_data["is_update"]:
            # Update problem description
            if analysis_data["problem_text"]:
                self.conversation_agent.conversation.problem_text = analysis_data["problem_text"]
            # Update background information
            self.conversation_agent.conversation.background_info.update(analysis_data["background_info"])
            # Add new solutions
            for solution in analysis_data["solutions"]:
                self.conversation_agent.add_solution(solution["title"], solution["subtitle"])
        
        # Step 3: Find relevant documents based on all aspects of the conversation
        query_text = f"""
        Problem: {self.conversation_agent.conversation.problem_text}
        Background: {json.dumps(self.conversation_agent.conversation.background_info)}
        Solutions: {", ".join([f"{s['title']} - {s['subtitle']}" for s in analysis_data["solutions"]])}
        """
        similar_docs = self.embedding_manager.find_similar(query_text, top_k=20)
        
        # Step 4: Generate insights from documents
        # Separate documents by type
        text_docs = [(doc, score) for doc, score in similar_docs if doc.metadata.get("file_type") == "text"]
        csv_docs = [(doc, score) for doc, score in similar_docs if doc.metadata.get("file_type") == "csv"]
        
        # Generate insights from text documents
        if text_docs:
            text_insights = self.text_insight_chain.invoke(
                current_problem=self.conversation_agent.conversation.problem_text,
                current_background=json.dumps(self.conversation_agent.conversation.background_info),
                current_solutions=json.dumps([
                    {"title": s.title, "subtitle": s.subtitle}
                    for s in self.conversation_agent.conversation.solutions
                ]),
                previous_insights=json.dumps([
                    {"text": i.text, "sources": i.sources}
                    for i in self.conversation_agent.conversation.insights
                ]),
                documents=json.dumps([doc.page_content for doc, _ in text_docs])
            )
            text_insight_data = json.loads(text_insights)
            
            # Add text-based insights
            for insight in text_insight_data["insights"]:
                self.conversation_agent.add_insight(
                    text=insight["text"],
                    sources=insight["sources"],
                    vega_lite_spec=insight.get("vega_lite_spec") if insight["needs_visualization"] else None
                )
        
        # Generate insights from CSV documents
        if csv_docs:
            csv_agent = self._create_csv_insight_agent(csv_docs)
            csv_insight_prompt = f"""You are an AI assistant analyzing a continuous medical dialogue between a doctor and a patient.
            Your task is to analyze the provided CSV data and generate new insights that are relevant to the current medical consultation.
            
            Important context:
            1. This is part of an ongoing medical consultation
            2. You have access to:
               - The current problem description: {self.conversation_agent.conversation.problem_text}
               - The current background information: {json.dumps(self.conversation_agent.conversation.background_info)}
               - The current solutions: {json.dumps([{"title": s.title, "subtitle": s.subtitle} for s in self.conversation_agent.conversation.solutions])}
               - Previously generated insights: {json.dumps([{"text": i.text, "sources": i.sources} for i in self.conversation_agent.conversation.insights])}
            3. You should:
               - Generate new insights that are strictly based on the CSV data
               - Ensure new insights are not similar to previously generated ones
               - For each insight, specify which CSV files were used as sources
               - If an insight would be better conveyed through a visualization, generate a Vega-Lite specification
               - For visualization insights, provide a brief explanation of what the chart shows
            
            Analyze the CSV data and return a JSON with the following structure:
            {{
                "insights": [
                    {{
                        "text": "Insight text or chart explanation",
                        "sources": ["csv_file1", "csv_file2", ...],
                        "needs_visualization": true/false,
                        "vega_lite_spec": {{
                            // Vega-Lite specification if needs_visualization is true
                            // null if needs_visualization is false
                        }}
                    }}
                ]
            }}
            """
            
            csv_insights = csv_agent.invoke(csv_insight_prompt)
            csv_insight_data = json.loads(csv_insights)
            
            # Add CSV-based insights
            for insight in csv_insight_data["insights"]:
                self.conversation_agent.add_insight(
                    text=insight["text"],
                    sources=insight["sources"],
                    vega_lite_spec=insight.get("vega_lite_spec") if insight["needs_visualization"] else None
                )
        
        # Return updated conversation state
        # return {
        #     "problem_text": self.conversation_agent.conversation.problem_text,
        #     "background_info": self.conversation_agent.conversation.background_info,
        #     "solutions": [
        #         {"title": s.title, "subtitle": s.subtitle}
        #         for s in self.conversation_agent.conversation.solutions
        #     ],
        #     "insights": [
        #         {
        #             "text": i.text,
        #             "sources": i.sources,
        #             "vega_lite_spec": i.vega_lite_spec
        #         }
        #         for i in self.conversation_agent.conversation.insights
        #     ]
        # }
        return self.conversation_agent.conversation
