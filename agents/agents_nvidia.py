import warnings
warnings.filterwarnings('ignore')

# Import crew librery
from crewai import Agent, Task, Crew

# Connecting to any LLM (NVIDIA)

from dotenv import load_dotenv
from crewai import Agent
from langchain_nvidia_ai_endpoints import ChatNVIDIA

# Import gradio librery
import gradio as gr


import os

load_dotenv()

nvapi_key=os.environ.get("NVIDIA_API_KEY")

# Check for the available Nvidia LLM model
llm_nvidia = ChatNVIDIA(model="meta/llama3-8b-instruct", nvidia_api_key=nvapi_key)

# Create Agents

planner = Agent(
    role="Planificador de Clase",  # Content Planner -> Planificador de Clase de Biología de un grupo de niños de 13 años
    goal="Planificar contenido atractivo y factualmente preciso sobre {topic}", 
    backstory="Estás trabajando en la planificación de una clase de biología para niños de 13 años en un instituto de España"
             "El tema: {topic} debe de estar en el contexto de una clase de Biología para niños de 13 años. "
             "Cuando el tema: {topic} no tenga que ver con una clase de Biología para niños de 13 años responde que NO HAY PLANIFICACIÓN. " 
             "Recopilas información que ayuda a la creación de la lección. "
             "Tu trabajo es la base para que el Redactor de Contenido escriba "
             "El texto debe de estar escrito en español y ser adecuado y atractivo para niños de 13 años. ",  
    llm=llm_nvidia,
    allow_delegation=False,
    verbose=True
)

writer = Agent(
    role="Redactor de Contenido",  
    goal="Escribir el texto en español que se les va a dar a los niños en clase "
         "sobre el tema: {topic}",  
    backstory="Estás trabajando en la redacción de un texto en español para la clase de biología de niños de 13 años"
             "sobre el tema: {topic} que debe de estar en el contexto de una clase de Biología para niños de 13 años."
             "Basas tu redacción en el trabajo del "
             "Planificador de Clase, quien te proporciona un esquema "
             "y contexto relevante sobre el tema. Sigues los objetivos "
             "principales y la dirección del esquema, tal como lo proporciona "
             "el Planificador de Clase.", 
    llm=llm_nvidia,
    allow_delegation=False,
    verbose=True
)

editor = Agent(
    role="Editor",
    goal="Editar una publicación de la clase en un blog que asegurandose"
         " que el {topic} tiene que ver con una clase de Biología para niños de 13 años.",  
    backstory="Eres el editor del colegio que recibe una publicación de blog "
             "del Redactor de Contenido. Tu objetivo es revisar la "
             "publicación para asegurarte de que sigue las mejores prácticas "
             ", que proporciona datos correctos para niños de 13 años ", 
    llm=llm_nvidia,
    allow_delegation=False,
    verbose=True
)

# Create Task

plan = Task(
    description=(
        "1. Prioriza los conceptos clave sobre {topic}.\n"  
        "2. El público objetivo son alumnos de 13 años en una clase de biología.\n"  
        "3. Desarrolla un esquema de contenido detallado que incluya una introducción, puntos clave y una llamada a la acción."
    ),
    expected_output="Un documento completo de planificación de contenido "
                   "con un esquema, palabras clave SEO y recursos.",  
    agent=planner,
)

write = Task(
    description=(
        "1. Usa el plan de contenido para crear una publicación de blog atractiva para alumnos de 13 años en una clase de biología sobre {topic}.\n" 
        "2. Incorpora palabras clave SEO de forma natural.\n"  
        "3. Las secciones/subtítulos se nombran adecuadamente de manera atractiva.\n"  
        "4. Asegúrate de que la publicación esté estructurada con una introducción atractiva, un cuerpo perspicaz y una conclusión resumida.\n"  
        "5. Asegurate de que el contenido está en español.\n"  
        "6. Revisa la publicación para detectar errores gramaticales y asegurarse de que el contenido es apto para una clase de Biología de niños de 13 años.\n"  
    ),
    expected_output="Una publicación de blog bien escrita en formato markdown, "
                   "lista para su publicación, cada sección debe tener de 2 a 3 párrafos.",  
    agent=writer,
)

edit = Task(
    description=("Revisa la publicación de blog dada para detectar errores gramaticales "
                 "y asegurarse de que el contenido es apto para una clase de Biología de niños de 13 años."),  
    expected_output="Una publicación de blog bien escrita en formato markdown y en español, "
                   "lista para su publicación, cada sección debe tener de 2 a 3 párrafos.",  
    agent=editor
)

# Creating the Crew

crew = Crew(
    agents=[planner, writer, editor],
    tasks=[plan, write, edit]
)

def predict(text):
  return crew.kickoff(inputs={"topic": text})

title = "CLASE DE BIOLOGÍA PARA 1º DE LA ESO"

iface = gr.Interface(
  fn=predict, 
  inputs=[gr.Textbox(label="Tema", lines=3)],
  outputs='text',
  title=title,
  examples=[["Clasificación de los seres vivos"], ["La célula"]]
)

iface.launch(debug=True)