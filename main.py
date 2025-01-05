#Zehaan Walji
#Dec 11th, 2024
#Agent Crew File

#Imports:
import os
from crewai import Crew, Process
from agents import MedicalAgents
from tasks import MedicalTasks
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()



# You can define as many agents and tasks as you want in agents.py and tasks.py
class MedicalCrew:

    #Passing through all user input
    def __init__(self, query):
        self.query = query
        self.OpenAIGPT4 = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1500, #Adjust if the words are being cut for whatever reason # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY") # type: ignore
        )


    #Run function:
    def run(self):

        #Initialzing the agents and tasks that were created in agents.py and tasks.py
        agents = MedicalAgents()
        tasks = MedicalTasks()
        masterAgent = agents.masterAgent()
        symptomAnalysisAgent = agents.symptomAnalysisAgent()
        advisorAgent = agents.advisorAgent()
        riskAssesmentAgent = agents.riskAssesmentAgent()
        verificationAgent= agents.verificationAgent()
        userProficiencyAgent = agents.userProficiencyAgent()

        #Defining Tasks:
        classifySymptoms = tasks.classifySymptoms(
            agent= symptomAnalysisAgent,
            query= self.query
        )
        recommendProtocol = tasks.recommendProtocol(
            agent= advisorAgent,
            context= [classifySymptoms]
        )
        verifyRecommendation = tasks.verifyRecommendation(
            agent= verificationAgent,
            context= [recommendProtocol]

        )
        checkUserMedicalKnowledge= tasks.checkUserMedicalKnowledge(
            agent= userProficiencyAgent,
            query= self.query
        )
        escelateRisk = tasks.escelateRisk(
            agent= riskAssesmentAgent,
            context=[recommendProtocol]
        )

        userExplination = tasks.userExplination(
            agent= masterAgent,
            context= [classifySymptoms,recommendProtocol, verifyRecommendation]
        )

        



        # Define the crew of agents:
        crew = Crew(

            agents= [masterAgent, 
                    symptomAnalysisAgent, 
                    advisorAgent, 
                    riskAssesmentAgent,
                    verificationAgent,
                    userProficiencyAgent,
                    ],

            tasks=[classifySymptoms, 
                   recommendProtocol,
                   verifyRecommendation,
                    checkUserMedicalKnowledge,
                    escelateRisk,
                   userExplination,
                   ],


            verbose=True,
            process=Process.sequential,
            memory=True,
            manager_llm=self.OpenAIGPT4,
        )

        result = crew.kickoff()
        return result