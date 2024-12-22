#Zehaan Walji
#Dec 11th, 2024
#Agent Crew File



from dotenv import load_dotenv
import os
os.environ["OPENAI_API_KEY"] = "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"


# Load environment variables from .env
load_dotenv()

# Verify if the key is loaded
print("OPENAI_API_KEY:", os.getenv("OPENAI_API_KEY"))
print("Anthropic API Key:", os.getenv("ANTHROPIC_API_KEY"))



#Imports:
import os
from crewai import Crew
from textwrap import dedent
from agents import MedicalAgents
from tasks import MedicalTasks
from dotenv import load_dotenv
load_dotenv()





# You can define as many agents and tasks as you want in agents.py and tasks.py
class MedicalCrew:

    #Passing through all user input
    def __init__(self, query):

        #Initalzing variables
        self.query = query


    #Run function:
    def run(self):
        #Initialzing the agents and tasks that were created in agents.py and tasks.py
        agents = MedicalAgents()
        tasks = MedicalTasks()

        #Defining agents:
        masterAgent = agents.masterAgent()
        symptomAnalysisAgent = agents.symptomAnalysisAgent()
        advisorAgent = agents.advisorAgent()
        riskAssesmentAgent = agents.riskAssesmentAgent()
        verificationAgents= [agents.verificationAgentOne(), agents.verificationAgentTwo(), agents.verificationAgentThree()]
        #verificationAgentOne = agents.verificationAgentOne()
        #verificationAgentTwo = agents.verificationAgentTwo()
       #verificationAgentThree = agents.verificationAgentThree()

        #Defining Tasks:
        classifySymptoms = tasks.classifySymptoms(
            agent= symptomAnalysisAgent,
            query= self.query
        )

        recommendProtocol = tasks.recommendProtocol(
            agent= advisorAgent,
            query= self.query
        )

        verifyRecommendation = tasks.verifyRecommendation(
            agent= verificationAgents,
            query= self.query
        )

        escelateRisk = tasks.escelateRisk(
            agent= riskAssesmentAgent,
            query= self.query
        )

        userExplination = tasks.userExplination(
            agent= masterAgent,
            query= self.query
        )
        

        # Define the crew of agents:
        crew = Crew(

            agents= [masterAgent, 
                    symptomAnalysisAgent, 
                    advisorAgent, 
                    riskAssesmentAgent,
                    *verificationAgents
                    ],

            tasks=[classifySymptoms, 
                   recommendProtocol,
                   verifyRecommendation,
                   userExplination,
                   escelateRisk],


            verbose=True,
        )

        result = crew.kickoff()
        return result





#Main function to run crew:
#This is subject to change later on
if __name__ == "__main__":
    #Printing a basic welcome message
    print("## Welcome to the Medical Aid MAS ")
    print('-------------------------------')


#Basic query input
    query = input(
        dedent("""
      What issue are you currently facing?
    """))


    medicalCrew= MedicalCrew(query)

    result = medicalCrew.run()
    print("\n\n########################")
    print("## FINAL OVERVIEW")
    print("########################\n")
    print(result)