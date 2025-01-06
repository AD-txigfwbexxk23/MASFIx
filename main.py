import os
from crewai import Crew, Process
from agents import MedicalAgents
from tasks import MedicalTasks
from dotenv import load_dotenv

# Memory imports
from langchain_openai import ChatOpenAI
from mem0 import MemoryClient

load_dotenv()

class MedicalCrew:
    def __init__(self, query):
        self.query = query
        self.OpenAIGPT4 = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1500, # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY") # type: ignore
        )
        
        # Initialize memory client
        self.client = MemoryClient(api_key=os.environ.get("MEM0_API_KEY"))

    def fetch_past_queries(self):
        try:
            response = self.client.search(
                query="*",  # Fetch all past queries
                user_id="ZehaanWalji",
                agent_id="MedicalAgent",
                limit=10,
                sort="desc"
            )
            return response.get("results", [])  # Safely return results or an empty list
        except Exception as e:
            print(f"Error fetching past queries: {e}")
            return []

    def add_to_memory(self, query, messages):
        try:
            self.client.add(
                query=query,
                messages=messages,
                user_id="ZehaanWalji",
                agent_id="MedicalAgent"
            )
        except Exception as e:
            print(f"Error adding query to memory: {e}")

    def run(self):
        # Fetch past queries for context
        past_queries = self.fetch_past_queries()

        # Add the current query to memory
        self.add_to_memory(query=self.query, messages=[{"role": "user", "content": self.query}])

        # Initialize agents and tasks
        agents = MedicalAgents(memory_client=self.client)  # Pass memory client here
        tasks = MedicalTasks()
        masterAgent = agents.masterAgent()
        symptomAnalysisAgent = agents.symptomAnalysisAgent()
        advisorAgent = agents.advisorAgent()
        riskAssessmentAgent = agents.riskAssesmentAgent()
        verificationAgent = agents.verificationAgent()
        userProficiencyAgent = agents.userProficiencyAgent()

        # Define tasks
        classifySymptoms = tasks.classifySymptoms(
            agent=symptomAnalysisAgent,
            query=self.query,
            past_queries=past_queries
        )

        recommendProtocol = tasks.recommendProtocol(
            agent=advisorAgent,
            context=[classifySymptoms],
            past_queries=past_queries
        )

        verifyRecommendation = tasks.verifyRecommendation(
            agent=verificationAgent,
            context=[recommendProtocol]
        )

        checkUserMedicalKnowledge = tasks.checkUserMedicalKnowledge(
            agent=userProficiencyAgent,
            query=self.query
        )

        escalateRisk = tasks.escelateRisk(
            agent=riskAssessmentAgent,
            context=[recommendProtocol]
        )

        userExplanation = tasks.userExplination(
            agent=masterAgent,
            context=[classifySymptoms, recommendProtocol, verifyRecommendation],
            past_queries=past_queries
        )

        # Configure Crew
        crew = Crew(
            agents=[
                masterAgent,
                symptomAnalysisAgent,
                advisorAgent,
                riskAssessmentAgent,
                verificationAgent,
                userProficiencyAgent
            ],
            tasks=[
                classifySymptoms,
                recommendProtocol,
                verifyRecommendation,
                checkUserMedicalKnowledge,
                escalateRisk,
                userExplanation
            ],
            process=Process.sequential,
            memory=True,
            verbose=True,
            memory_config={
                "provider": "mem0",
                "config": {
                    "user_id": "ZehaanWalji",
                    "api_key": os.environ.get("MEM0_API_KEY"),
                },
            },
        )

        try:
            result = crew.kickoff()
            return result
        except Exception as e:
            print(f"Error during Crew execution: {e}")
            return {"error": str(e)}
