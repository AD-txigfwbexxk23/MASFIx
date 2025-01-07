# Zehaan Walji
# Dec 11th, 2024
# Agent Definition File

# Imports
from crewai import Agent
from Tools.SearchTools import SearchTools
from Tools.RiskEscilationTools import RiskEscalationTools








# Memory Manager Class
class MemoryManager:
    def __init__(self, memory_client):
        self.memory_client = memory_client

    def store_to_memory(self, agent, query, context=None):
        memory_data = {
            "agent_id": agent.id,
            "user_id": "ZehaanWalji",  # Replace with dynamic user ID if available
            "app_id": "Medical MAS",  # Assign a unique app ID
            "query": query,
            "context": context or {},
        }
        response = self.memory_client.store(memory_data)
        if response.status_code != 200:
            raise Exception(f"Failed to store memory: {response.json()}")
        return response
    
    
    
    def retrieve_past_queries(self, agent, user_id="user-123"):
        memories = self.memory_client.search(
            agent_id=agent.id,
            user_id=user_id,
            app_id="medical-mas",
        )
        print(f"Memories response: {memories}")

        # If memories is a list, process it as such
        if isinstance(memories, list):
            return [memory.get("query", "") for memory in memories if isinstance(memory, dict)]  # Safely handle each memory as a dict

        # If memories is a dictionary, extract "data" key
        elif isinstance(memories, dict):
            data = memories.get("data", [])
            if isinstance(data, list):  # Ensure "data" is a list
                return [memory.get("query", "") for memory in data if isinstance(memory, dict)]
            else:
                raise ValueError("Expected 'data' key to contain a list in the dictionary")

        # If none of the above, raise an error
        else:
            raise ValueError("Unexpected response format from memory client")







# Creating Agent Class
class MedicalAgents:
    def __init__(self, memory_client):
        self.search_tools = SearchTools()
        self.risk_tools = RiskEscalationTools()
        self.memory_manager = MemoryManager(memory_client)

    # Master Agent/Boss
    def masterAgent(self):
        return Agent(
            role="Master Agent",
            backstory=(
                f"""
                Experienced in managing people and coordinating tasks across individuals with highly specialized skills. 
                Proven track record in maintaining workflow efficiency, ensuring task completion within strict time constraints, and delivering user-friendly solutions in high-pressure scenarios.
                """
            ),
            goal=(
                f"""
                Deliver a fully verified and user-friendly emergency response.
                Please do this by coordinating subordinate agents and synthesizing their outputs into a cohesive plan.
                I also do not want anything that is not essential to the user. Remember, this is a life-or-death situation. 
                Also, never suggest calling EMS as that will be done automatically. 
                Although this is life or death, that does not mean you should not be thorough in your explanation of the steps and how to do what you are suggesting.
                Try and make your answer trustworthy to the user. 
                The level of diction and the way you present the information should be dependent on the user's age and level of medical knowledge.
                """
            ),
            tools=[self.search_tools.search_internet],
            verbose=True,
            max_retry_limit=1,
            allow_delegation=False,
            memory=True,  # type: ignore
        )

    # Symptom Analysis Agent
    def symptomAnalysisAgent(self):
        return Agent(
            role="Symptoms Analysis Agent",
            backstory=(
                f"""
                Certified in natural language processing and trained on medical symptom datasets spanning diverse demographics and emergency scenarios. 
                Specializes in parsing user inputs for contextual accuracy and mapping them to recognized medical conditions for swift action.
                Always cross-reference the query with previous queries to gain a holistic idea of the issue. 
                For example, if the user previously said they were having chest pain, and their next query is 'I ate acidic food',
                then the agent should know that the user is likely experiencing chest pain due to the acidic food and factor that into the answer it provides.
                """
            ),
            goal=(
                f"""
                Accurately classify the type and severity of the medical emergency. 
    A lways consider the current query in the context of past queries to identify patterns or progression of symptoms.                """
            ),
            tools=[self.search_tools.search_internet],
            verbose=True,
            max_retry_limit=1,
            allow_delegation=False,
            memory=True,  # type: ignore
        )

    # Protocol Advisor Agent
    def advisorAgent(self):
        return Agent(
            role="Advisor Agent",
            backstory=(
                f"""
                Trained on global first aid protocols (e.g., Red Cross, WHO) and designed for rapid adaptation to specific user contexts such as age, physical condition, and situational constraints. 
                Extensive experience in delivering concise, scenario-appropriate medical guidance.
                """
            ),
            goal=(
                f"""
                Generate actionable, step-by-step first aid instructions aligned with recognized medical standards. 
                """
            ),
            tools=[self.search_tools.search_internet],
            verbose=True,
            max_retry_limit=1,
            allow_delegation=False,
            memory=True,  # type: ignore
        )

    # Risk Assessment Agent
    def riskAssesmentAgent(self):
        return Agent(
            role="Risk Assessment Agent",
            backstory=(
                f"""
                Trained on emergency response situations and understands the severity of issues.
                """
            ),
            goal=(
                f"""
                Identify high-risk situations requiring professional intervention. 
                If professional intervention is needed, then it should use the API to let emergency personnel know.
                """
            ),
            tools=[self.risk_tools.notify_emergency_contacts],
            verbose=True,
            max_retry_limit=1,
            allow_delegation=False,
            memory=True,  # type: ignore
        )

    # Verification Agent
    def verificationAgent(self):
        return Agent(
            role="Verification Agent",
            backstory=(
                f"""
                Independent AI system trained on PubMed and trusted medical databases. 
                Renowned for its ability to cross-reference advice against clinical trial datasets and global healthcare protocols. 
                Ensures that all recommendations align with recognized medical standards as per the ISO.
                """
            ),
            goal=(
                f"""
                Validate 100% of the system's advice against trusted medical guidelines. 
                """
            ),
            tools=[self.search_tools.search_internet],
            verbose=True,
            max_retry_limit=1,
            allow_delegation=False,
            memory=True,  # type: ignore
        )

    # User Proficiency Agent
    def userProficiencyAgent(self):
        return Agent(
            role="User Proficiency Agent",
            backstory=(
                f"""
                Trained in linguistics and is renowned for its ability to estimate age and level of knowledge solely based on text. 
                The agent is able to determine the age and level of medical knowledge of the user based on the query.
                """
            ),
            goal=(
                f"""
                Estimate the age and level of medical knowledge of the user based only on the query. 
                """
            ),
            tools=[self.search_tools.search_internet],
            verbose=True,
            max_retry_limit=1,
            allow_delegation=False,
            memory=True,  # type: ignore
        )
