from typing import List, Dict, Any, Tuple
from agent.models import Conversation, Insight
from agent.conversation_agent import ConversationAgent
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import HumanMessage, SystemMessage, Document
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import pandas as pd
import json
import io

from agent.Azure_STT import AzureSpeechToText
from agent.embedding_manager import EmbeddingManager

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
        self.conversation_analysis_prompt = None
        self.text_insight_prompt = None
        
        # Initialize agents for different tasks
        self.conversation_analysis_chain = self._create_conversation_analysis_chain()
        self.text_insight_chain = self._create_text_insight_chain()
        
    def _create_conversation_analysis_chain(self):
        """Create a chain for analyzing and updating problem description, background information, and solutions."""
        print("\nCreating conversation analysis chain...")
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
                  - IMPORTANT: The problem description MUST be a single, concise line that captures the core issue
                  - Do not include background information or details in the problem description
                  - Keep it focused on the main medical condition or symptom
               
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
                "problem_text": "The problem description (updated or new) - MUST be a single line",
                "background_info": [{"key": "value"}], # Only include new or updated background information - This is a list of dictionaries with key-value pairs
                "solutions": [{"title": "Solution title", "subtitle": "Solution description"}, ...],  # Only include new solutions
                "is_update": true/false  # Indicates if any aspect was updated
            }
            """),
            HumanMessagePromptTemplate.from_template("""Current problem description: {current_problem}
                                                        Current background information: {current_background}
                                                        Current solutions: {current_solutions}
                                                        New transcription: {transcribed_text}""")
        ])
        
        # Store the prompt template for later use
        self.conversation_analysis_prompt = prompt
        
        # Create the chain using the new pattern
        chain = (
            {"transcribed_text": RunnablePassthrough(), 
             "current_problem": RunnablePassthrough(),
             "current_background": RunnablePassthrough(),
             "current_solutions": RunnablePassthrough()}
            | prompt
            | self.conversation_agent.llm
            | StrOutputParser()
        )
        
        print("Conversation analysis chain created successfully")
        return chain
    
    def _create_text_insight_chain(self):
        """Create a chain for generating insights from text documents."""
        print("\nCreating text insight chain...")
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an AI assistant analyzing a continuous medical dialogue between a doctor and a patient.
            Your task is to analyze the provided text documents and generate insights that are relevant to the current medical consultation to help the doctor make better decisions.
            
            Important context:
            1. This is part of an ongoing medical consultation
            2. You have access to:
               - The current problem description
               - The current background information
               - The current solutions
               - Previously generated insights
               - The documents that are relevant to the current medical consultation
            3. You should:
               - Generate insights that are strictly based on the provided documents.
               - You can use the previous insights as a reference to either:
                 a. Keep and improve upon them if they are still relevant
                 b. Replace them with new insights if they are no longer relevant
                 c. Combine multiple previous insights into a single, more comprehensive insight
               - For each insight, specify which documents were used as sources. These documents are from the documents list that I will provide you with.
               - If an insight would be better conveyed through a visualization, generate a Vega-Lite specification
               - You should at least generate one insight that has a visualization.
               - For visualization insights, provide a brief explanation of what the chart shows
            
            Return a JSON with the following structure:
            {
                "insights": [
                    {
                        "text": "Insight text or chart explanation",
                        "sources": ["DOCUMENT NAME"], # A list of document names that were used to generate the insight. These documents are from the documents list that I will provide you with.
                        "needs_visualization": true/false,
                        "vega_lite_spec": {
                            // Vega-Lite specification if needs_visualization is true
                            // null if needs_visualization is false
                        }
                    }
                ]
            }
            """),
            HumanMessagePromptTemplate.from_template("""Current problem description: {current_problem}
            Current background information: {current_background}
            Current solutions: {current_solutions}
            Previous insights: {previous_insights}
            Documents: {documents}
            """)
        ])
        
        # Store the prompt template for later use
        self.text_insight_prompt = prompt
        
        # Create the chain using the new pattern
        chain = (
            {"current_problem": RunnablePassthrough(),
             "current_background": RunnablePassthrough(),
             "current_solutions": RunnablePassthrough(),
             "previous_insights": RunnablePassthrough(),
             "documents": RunnablePassthrough()}
            | prompt
            | self.conversation_agent.llm
            | StrOutputParser()
        )
        
        print("Text insight chain created successfully")
        return chain
    
    def _create_csv_insight_agent(self, csv_docs: List[Tuple[Document, float]]) -> AgentExecutor:
        """Create an agent for generating insights from CSV documents."""
        print("\nCreating CSV insight agent...")
        # Create a DataFrame from all CSV documents
        dfs = []
        for doc, _ in csv_docs:
            file_path = doc.metadata.get('source_file', 'unknown')
            print(f"Processing CSV document: {file_path}")
            df = pd.read_csv(file_path)
            print(f"DataFrame columns: {df.columns}")
            dfs.append(df)
        
        print(f"Total number of DataFrames: {len(dfs)}")
        # Create a pandas DataFrame agent
        agent = create_pandas_dataframe_agent(
            llm=self.conversation_agent.llm,
            dfs=dfs,
            verbose=True,
            agent_type=AgentType.OPENAI_FUNCTIONS
        )
        print("CSV insight agent created successfully")
        return agent
    
    # def process_audio(self, audio_path: str) -> Dict[str, Any]:
    def process_audio(self, transcribed_text: str) -> Conversation:
        """Process an audio file and update the conversation."""
        print("\n=== Starting process_audio ===")
        print(f"Input transcription: {transcribed_text[:100]}...")  # Show first 100 chars
        
        # Step 2: Analyze conversation state and update
        print("\n--- Step 2: Analyzing conversation state ---")
        print("Current conversation state:")
        print(f"Problem: {self.conversation_agent.conversation.problem_text}")
        print(f"Background info: {json.dumps(self.conversation_agent.conversation.background_info)}")
        print(f"Current solutions: {json.dumps([{'title': s.title, 'subtitle': s.subtitle} for s in self.conversation_agent.conversation.solutions])}")
        
        # Prepare the input for the chain
        chain_input = {
            "transcribed_text": transcribed_text,
            "current_problem": self.conversation_agent.conversation.problem_text or "No current problem description.",
            "current_background": json.dumps(self.conversation_agent.conversation.background_info) or "No current background information.",
            "current_solutions": json.dumps([
                {"title": s.title, "subtitle": s.subtitle}
                for s in self.conversation_agent.conversation.solutions
            ]) or "No current solutions."
        }
        
        # Print the formatted prompt for debugging
        print("\nFormatted prompt being sent to LLM:")
        formatted_messages = self.conversation_analysis_prompt.format_messages(**chain_input)
        for message in formatted_messages:
            print(f"{message.type}: {message.content}")
        
        print("\nInvoking analysis chain...")
        analysis = self.conversation_analysis_chain.invoke(chain_input)
        print("\nAnalysis chain output:")
        print(analysis)
        
        # Extract the JSON from the response
        try:
            # Find the JSON content between ```json and ``` markers
            json_start = analysis.find("```json\n") + 8
            json_end = analysis.find("\n```", json_start)
            if json_start > 7 and json_end > json_start:
                json_str = analysis[json_start:json_end]
                analysis_data = json.loads(json_str)
            else:
                # If no JSON markers found, try parsing the entire response
                analysis_data = json.loads(analysis)
        except json.JSONDecodeError as e:
            print(f"Error parsing analysis response: {e}")
            print("Raw response:", analysis)
            raise
        
        # Update conversation with analysis only if there are updates
        if analysis_data["is_update"]:
            print("\n--- Updating conversation with analysis ---")
            # Update problem description
            if analysis_data["problem_text"]:
                print(f"Updating problem text to: {analysis_data['problem_text']}")
                self.conversation_agent.conversation.problem_text = analysis_data["problem_text"]
            # Update background information
            print(f"Updating background info with: {json.dumps(analysis_data['background_info'])}")
            # Convert list of single-item dictionaries to a single dictionary
            background_dict = {}
            for item in analysis_data["background_info"]:
                background_dict.update(item)
            self.conversation_agent.conversation.background_info.update(background_dict)
            
            # Add new solutions
            print(f"Adding new solutions: {json.dumps(analysis_data['solutions'])}")
            for solution in analysis_data["solutions"]:
                self.conversation_agent.add_solution(solution["title"], solution["subtitle"])
        
        # Step 3: Find relevant documents
        print("\n--- Step 3: Finding relevant documents ---")
        query_text = f"""
        Problem: {self.conversation_agent.conversation.problem_text}
        Background: {json.dumps(self.conversation_agent.conversation.background_info)}
        Solutions: {", ".join([f"{s['title']} - {s['subtitle']}" for s in analysis_data["solutions"]])}
        """
        print(f"Search query: {query_text[:200]}...")  # Show first 200 chars
        
        similar_docs = self.embedding_manager.find_similar(query_text, top_k=5)
        print(f"Found {len(similar_docs)} similar documents")
        
        # Step 4: Generate insights from documents
        print("\n--- Step 4: Generating insights ---")
        # Separate documents by type
        text_docs = [(doc, score) for doc, score in similar_docs if doc.metadata.get("file_type") == "text"]
        csv_docs = [(doc, score) for doc, score in similar_docs if doc.metadata.get("file_type") == "csv"]
        print(f"Found {len(text_docs)} text documents and {len(csv_docs)} CSV documents")
        
        # Initialize insight lists
        text_insights = []
        csv_insights = []
        
        # Generate insights from text documents
        if text_docs:
            print("\nProcessing text documents...")
            print("Invoking text insight chain...")
            
            # Format documents with their source files
            formatted_documents = []
            for doc, _ in text_docs:
                source_file = doc.metadata.get('source_file', 'unknown')
                formatted_documents.append({
                    "content": doc.page_content,
                    "source": source_file
                })
            
            chain_input = {
                "current_problem": self.conversation_agent.conversation.problem_text,
                "current_background": json.dumps(self.conversation_agent.conversation.background_info),
                "current_solutions": json.dumps([
                    {"title": s.title, "subtitle": s.subtitle}
                    for s in self.conversation_agent.conversation.solutions
                ]),
                "previous_insights": json.dumps([
                    {"text": i.text, "sources": i.sources}
                    for i in self.conversation_agent.conversation.insights
                ]),
                "documents": json.dumps(formatted_documents)
            }
            
            # Print the formatted prompt for debugging
            print("\nFormatted prompt being sent to LLM:")
            formatted_messages = self.text_insight_prompt.format_messages(**chain_input)
            for message in formatted_messages:
                print(f"{message.type}: {message.content}")
            
            print("\nInvoking text insight chain...")
            text_insights = self.text_insight_chain.invoke(chain_input)
            print("Text insight chain output:")
            print(text_insights)
            
            # Extract the JSON from the response
            try:
                # Find the JSON content between ```json and ``` markers
                json_start = text_insights.find("```json\n") + 8
                json_end = text_insights.find("\n```", json_start)
                if json_start > 7 and json_end > json_start:
                    json_str = text_insights[json_start:json_end]
                    text_insight_data = json.loads(json_str)
                else:
                    # If no JSON markers found, try parsing the entire response
                    text_insight_data = json.loads(text_insights)
            except json.JSONDecodeError as e:
                print(f"Error parsing text insight response: {e}")
                print("Raw response:", text_insights)
                raise
            
            print(f"Adding {len(text_insight_data['insights'])} text-based insights")
            
            # Store text-based insights
            text_insights = []
            for insight in text_insight_data["insights"]:
                print(f"Adding insight: {insight['text'][:100]}...")
                text_insights.append(Insight(
                    text=insight["text"],
                    sources=insight["sources"],
                    vega_lite_spec=insight.get("vega_lite_spec") if insight["needs_visualization"] else None
                ))
        
        # Generate insights from CSV documents
        if csv_docs:
            print("\nProcessing CSV documents...")
            csv_agent = self._create_csv_insight_agent(csv_docs)
            csv_insight_prompt = f"""You are an AI assistant analyzing a continuous medical dialogue between a doctor and a patient.
            Your task is to analyze the provided CSV data and generate insights that are relevant to the current medical consultation.
            
            Important context:
            1. This is part of an ongoing medical consultation
            2. You have access to:
               - The current problem description: {self.conversation_agent.conversation.problem_text}
               - The current background information: {json.dumps(self.conversation_agent.conversation.background_info)}
               - The current solutions: {json.dumps([{"title": s.title, "subtitle": s.subtitle} for s in self.conversation_agent.conversation.solutions])}
               - Previously generated insights: {json.dumps([{"text": i.text, "sources": i.sources} for i in self.conversation_agent.conversation.insights])}
            3. You should:
               - Generate insights that are strictly based on the CSV data
               - You can use the previous insights as a reference to either:
                 a. Keep and improve upon them if they are still relevant
                 b. Replace them with new insights if they are no longer relevant
                 c. Combine multiple previous insights into a single, more comprehensive insight
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
            
            print("Invoking CSV agent...")
            csv_insights = csv_agent.invoke(csv_insight_prompt)
            print("CSV agent output:")
            print(csv_insights)
            
            csv_insight_data = json.loads(csv_insights)
            print(f"Adding {len(csv_insight_data['insights'])} CSV-based insights")
            
            # Store CSV-based insights
            csv_insights = []
            for insight in csv_insight_data["insights"]:
                print(f"Adding insight: {insight['text'][:100]}...")
                csv_insights.append(Insight(
                    text=insight["text"],
                    sources=insight["sources"],
                    vega_lite_spec=insight.get("vega_lite_spec") if insight["needs_visualization"] else None
                ))
        
        # Combine all insights and update the conversation
        all_insights = text_insights + csv_insights
        self.conversation_agent.conversation.insights = all_insights
        
        print("\n=== Completed process_audio ===")
        print("Conversation state:")
        print(self.conversation_agent.conversation)
        

        return self.conversation_agent.conversation
