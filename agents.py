#Zehaan Walji
#Dec 11th, 2024
#Agent Definition File



#Imports
import os
from crewai import Agent
from textwrap import dedent
from Tools.SearchTools import SearchTools
from Tools.RiskEscilationTools import RiskEscalationTools
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic



#Notes:
"""
Goal:
- Explain step-by-step how to solve a given medical related issue to someone experiencing crisis 

Captain/Manager/Boss:
- Expert paramedic; oversees the operation
- Manages the workflow for subordinate agents and presents the final advice to the user
- Will need to identify the type of emergency, delegate tasks to specific employees, and communicate step by step what to do while inclduing automation transparency


Employees/Experts to hire:
- Symptom Analysis Agent: Classifies the severity of the issue
- Protocol Advisor Agent: Retrieve and adapt specific medical protocols based on the emergency type
- Trust Enhancer Agent: Improve the user’s confidence in the system and explain what each agent is doing
- Risk Assesment Agent: Assess risks and decide what solutions could be best
- Verification Agent (1), (2), (3): Checks the output with other AI systems


Notes:
- Agents should be results driven and have a clear goal in mind
- Role is their job title
- Goals should actionable
- Backstory should be their resume
"""




#Creating Agent Class
class MedicalAgents:


    def __init__(self):
        #Instanciating (Important for the file to run correctly)
        self.search_tools = SearchTools()
        self.risk_tools = RiskEscalationTools()

        #ChatGPT 3.5
        self.OpenAIGPT35 = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.7,
            max_tokens=1500, #Adjust if the words are being cut for whatever reason # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY") # type: ignore
        )
        #ChatGPT 4o
        self.OpenAIGPT4 = ChatOpenAI(
            model="gpt-4",
            temperature=0.7,
            max_tokens=1500, #Adjust if the words are being cut for whatever reason # type: ignore
            openai_api_key=os.environ.get("OPENAI_API_KEY") # type: ignore
        )
        #Claude
        self.Claude = ChatAnthropic(
            model="claude-2",  # Specify the Claude version # type: ignore
            max_tokens_to_sample=1500, # type: ignore
            temperature=0.7,
            anthropic_api_key=os.environ.get("ANTHROPIC_API_KEY") # type: ignore
        )



#Master Agent/Boss
    def masterAgent(self):
        return Agent(

            role="Master Agent",
            backstory=dedent(f"""
                             
            Experienced in managing people and coordinating tasks across indivudals with highly specialized skills. 
            Proven track record in maintaining workflow efficiency, ensuring task completion within strict time constraints, and delivering user-friendly solutions in high-pressure scenarios.
                            
                              """),
                        
            goal=dedent(f"""
                        
            Deliver a fully verified and user-friendly emergency response.
            Please do this by coordinating subordinate agents and synthesizing their outputs into a cohesive plan.
            I also do not want anything that is not essential to the user. Remember, this is a life-or-death situation. 
            Also, never suggest calling EMS as that will be done automatically. 
            Although this is life or death, that does not mean you should not be thorough in your explanation of the steps and how to do what you are suggesting.
            Also try and make your answer trustworthy to the user. 
                                                
                        
                        """),

            tools=[],

            verbose=True,
            llm=self.OpenAIGPT35, #Using ChatGPT 3.5 for now but I will update to ChatGPT4 later (The bill is quite expensive per run otherwise)
        )




#Symptom Analysis Agent
    def symptomAnalysisAgent(self):
        return Agent(
            role="Symptoms Analysis Agent",
            backstory=dedent(f"""
                             
            Certified in natural language processing and trained on medical symptom datasets spanning diverse demographics and emergency scenarios. 
            Specializes in parsing user inputs for contextual accuracy and mapping them to recognized medical conditions for swift action.
                             
                             """),
            goal=dedent(f"""
                        
            Accurately classify the type and severity of the medical emergency.
                                               
                        """),

            tools=[self.search_tools.search_internet],

            verbose=True,
            llm=self.OpenAIGPT35,
        )
    




#Protocol Advisor Agent
    def advisorAgent(self):
        return Agent(
            role="Advisor Agent",
            backstory=dedent(f"""
                             
            Trained on global first aid protocols (e.g., Red Cross, WHO) and designed for rapid adaptation to specific user contexts such as age, physical condition, and situational constraints. 
            Extensive experience in delivering concise, scenario-appropriate medical guidance.
                             
                             """),
            goal=dedent(f"""
                        
            Generate actionable, step-by-step first aid instructions aligned with recognized medical standards. 
                                               
                        """),

            tools=[self.search_tools.search_internet],

            verbose=True,
            llm=self.OpenAIGPT35,

        )


#Risk Assesment Agent
    def riskAssesmentAgent(self):
        return Agent(
            role="Risk Assesment Agent",
            backstory=dedent(f"""Trained on emergency response sitations and understand the severity of isues"""),
            goal=dedent(f"""
                        
            Identify high-risk situations requiring professional intervention. If professional intervention is needed then it should use the API to let emergency personel know.

                        
                        """),

            tools=[
                self.risk_tools.notify_emergency_contacts],


            verbose=True,
            llm=self.OpenAIGPT35,
        )





#Verification Agent (1):
#GPT3.5
#This agent is to comply with current ISO standards
    def verificationAgentOne(self):
        return Agent(
            role="Verification Agent",
            backstory=dedent(f"""
                             
            Independent AI system trained on PubMed and trusted medical databases. 
            Renowned for its ability to cross-reference advice against clinical trial datasets and global healthcare protocols. 
            Ensures that all recommendations align with recognized medical standards as per the ISO.
                             
                             
                             """),
            goal=dedent(f"""
                        
            Validate 100% of the system's advice against trusted medical guidelines. 
            Flag inconsistencies or errors within 3 seconds of receiving the proposed response.
                        
                        
                        """),

            tools=[self.search_tools.search_internet],

            verbose=True,
            llm=self.OpenAIGPT35,
        )





#Verification Agent (2):
#GPT 4o
    def verificationAgentTwo(self):
        return Agent(
            role="Verification Agent",
            backstory=dedent(f"""
                             
            Trained on a specialized medical LLM (e.g., OpenHermes by Ollama) for cross-verifying medical advice. 
            Adept at identifying deviations from globally accepted first aid protocols and offering corrections to improve accuracy.
                             
                             """),
            goal=dedent(f"""
                        
            Independently cross-check the advice provided by other agents. 
            Ensure all outputs align with global medical standards and highlight any discrepancies.
                        
                        """),

            tools=[self.search_tools.search_internet],

            verbose=True,
            llm=self.OpenAIGPT4, #Changing the LLM so that way the checks and balances system is using different networks
            #This should in theory help validte information better and prevent hallucination  
        )





#Verification Agent (3):
#Claude
    def verificationAgentThree(self):
        return Agent(
            role="Verification Agent",
            backstory=dedent(f"""
                             
            A specialized instance of OpenAI GPT-4, fine-tuned for medical guideline validation. 
            Focused on rapid response auditing and ensuring advice adheres to the highest standards of medical safety.
                             
                             """),
            goal=dedent(f"""
                        
            Serve as the final layer of validation, ensuring the advice meets all safety and accuracy standards. 

                        """),

            tools=[self.search_tools.search_internet],

            verbose=True,
            llm=self.OpenAIGPT35,#NEED TO FIX
        )