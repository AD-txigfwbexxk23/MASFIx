import streamlit as st
import openai 
from main import MedicalCrew
from imageGeneration import getImage
from dotenv import load_dotenv
import os
from pprint import pprint
import pyttsx3
from gtts import gTTS





class MedicalMASUI:

    #Initializing chatGPT and text to speech stuff (TEXT TO SPEECH WAS INITALIZED IN METHOD FOR NOW; NEED TO FIGURE OUT WHY)
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY") 
    openai.api_key = api_key








    #General parameter initialization for the class
    #Ensures that nothing is empty and that the code runs as it should
    def __init__(self):
        """
        Initialize session state variables.
        """
        if "query" not in st.session_state:
            st.session_state.query = ""

        if "generating" not in st.session_state:
            st.session_state.generating = False

        if "result" not in st.session_state:
            st.session_state.result = None











    #Taking in the query and striping it of any whitespace
    #Calling the medical crew function with the query
    def generate_medical_aid(self, query):
        """
        Initialize MedicalCrew with the required query and run it.
        """
        if not query.strip():
            st.warning("⚠️ Please enter a valid query before generating a solution.")
            return None

        crew = MedicalCrew(query=query)

        try:
            result = crew.run()
            return result
        
        except Exception as e:
            st.error(f"An error occurred: {e}")
            return None











    #Displaying the results
    def display_results(self, result):


        """
        Display agent outputs in a vertical layout with collapsible sections.
        """
        #If there is no result do nothing
        if not result:
            return #Do nothing

        #Getting all the task outputs
        tasks_output = getattr(result, 'tasks_output', [])
        if not isinstance(tasks_output, list):
            st.warning("⚠️ tasks_output is not valid. Please check the backend.")
            tasks_output = []



        # Correct Agent Mapping
        agent_roles = [
            ("Symptoms Analysis Agent", tasks_output[0] if len(tasks_output) > 0 else "No output provided by this agent."),
            ("Advisor Agent", tasks_output[1] if len(tasks_output) > 1 else "No output provided by this agent."),
            ("Verification Agent", tasks_output[2] if len(tasks_output) > 2 else "No output provided by this agent."),
        ]



        st.success("Your Medical Solution is Ready!")
        st.write("## **Agent Outputs and Thoughts**")












        # Master Agent Output
        master_agent_output = tasks_output[3] if len(tasks_output) > 3 else "No output provided by this agent."
        st.header("Master Agent")



        #GENERATING IMAGE AND WRAPPING IT AROUND THE MASTER AGENT TEXT
        #Initial message for chatGPT to have some context
        message = [{"role" : "assistant" , "content" : """
            You need to take in the list of advice and return short search query to find an image based on the response.
            I want nothing else in your output other then the sentence.
        """}]

        pprint(vars(st.session_state.result))
        print("\n")

        #Accessing the raw information and converting it into a string
        information=tasks_output[3] # type: ignore
        information= str(information)


        #Adding that information to the message for GPT
        message.append({"role": "user", "content": information})
        chat_response = openai.chat.completions.create(model="gpt-4o-mini",messages=message) # type: ignore


        #Getting the first reply
        reply = chat_response.choices[0].message.content
        print(reply)


        #Using the run method to generate the image links
        imageURL= getImage(str(reply))
        print(imageURL)
        if imageURL:
            imageURL= imageURL[0] 
        else:
            st.error("No image is available for this query")

        if imageURL:
            st.markdown(
                f"""
                    <div style="position: relative; overflow: hidden; line-height: 1.6;">
                        <img src="{imageURL}" alt="Generated Image"
                            style="float: left; width: 300px; height: auto; margin-right: 20px; margin-bottom: 10px; border-radius: 10px;">
                        <p style="text-align: justify;">{str(master_agent_output).replace('.', '.<br>')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
            )
        else:
            st.markdown(f"<p>{str(master_agent_output).replace('.', '.<br>')}</p>", unsafe_allow_html=True)



        # Display other agents
        for role, agent_output in agent_roles:
            with st.expander(f"{role}"):
                # Ensure the numbers and content stay on the same line
                formatted_output = str(agent_output).replace(".\n", ". ")
                st.markdown(f"<p>{formatted_output}</p>", unsafe_allow_html=True)



        # Risk Assessment Message
        #Note: Padding has been fixed so it no longer overlaps
        st.markdown(
            """
            <div style='background-color: #1b4332; color: #ffffff; padding: 15px; border-radius: 5px; margin-top: 20px; margin-bottom: 20px; text-align: center;'>
            <strong>Emergency personnel have been notified. Stay calm and await assistance.</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )


    
        #If the speak button is pressed
        if st.button("Speak"):
            #Creating a text to speech audo file
            speech= str(tasks_output[3])#Getting the master agent output as a string
            tts = gTTS(speech)#Uploading the string so it can be read as an mp3 file
            tts.save("output.mp3")
            #Playing the audio
            st.audio("output.mp3")
                    













    def run(self):
        """
        Main method to run the Streamlit UI.
        """
        #Setting the page title and icon as well as the sidebar title
        st.set_page_config(page_title="Medical MAS", page_icon="🩹")
        st.sidebar.title("Medical MAS Control Panel")
        st.sidebar.text("Use this panel to interact with the system.")

        # Input query
        query = st.sidebar.text_input(
            "Enter a medical query:", value=st.session_state.query, key="query"
        )

        # Generate button
        if st.sidebar.button("Generate Solution"):
            st.session_state.generating = True

        # Main layout
        if st.session_state.query.strip() == "":
            # Display MAS information in the main area
            st.title("Welcome to the Medical MAS System")
            st.markdown(
                """
                **What is a Multi-Agent System (MAS)?**
                
                A Multi-Agent System (MAS) is a system where multiple agents (software entities) collaborate, communicate, and solve tasks together. Each agent has its own role and knowledge base, contributing to a shared goal. In this medical MAS, different agents analyze symptoms, provide medical advice, and verify data, helping to create a comprehensive solution for your medical query.
                """
            )
        elif st.session_state.generating:
            result = self.generate_medical_aid(st.session_state.query)
            #Setting the st.session_state in streamlit equal to the raw output from crewAI
            st.session_state.result = result
            st.session_state.generating = False

        self.display_results(st.session_state.result)









if __name__ == "__main__":
    ui = MedicalMASUI()
    ui.run()



# Things to fix (try to do before the 6th):
# 1) Implemntation of Claude
# 2) Ability to ask follow up questions
# 3) Optionally facial recognition


# I am making a change