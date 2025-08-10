from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.schema import StrOutputParser

load_dotenv()

# LLM setup
llm = ChatOpenAI(temperature=0.7, model="gpt-4o-mini")

# Step 1 → Pet Name Generator
name_prompt = PromptTemplate(
    input_variables=["animal_type", "animal_colour"],
    template="Generate a unique and creative name for a {animal_colour} {animal_type}."
)

# Step 2 → Backstory Generator
story_prompt = PromptTemplate(
    input_variables=["pet_name", "animal_type", "animal_colour"],
    template="Write a short and fun backstory for a pet named {pet_name}, "
             "who is a {animal_colour} {animal_type}."
)

# Step 3 → Spanish Translation
translate_prompt = PromptTemplate(
    input_variables=["pet_story"],
    template="Translate the following text into Spanish:\n\n{pet_story}"
)

# Output parser to get plain string instead of AIMessage
parser = StrOutputParser()

# Build step-by-step chain
def generate_pet_info(animal_type, animal_colour):
    # Step 1 → get pet name
    pet_name = (name_prompt | llm | parser).invoke({
        "animal_type": animal_type,
        "animal_colour": animal_colour
    }).strip()

    # Step 2 → get backstory
    pet_story = (story_prompt | llm | parser).invoke({
        "pet_name": pet_name,
        "animal_type": animal_type,
        "animal_colour": animal_colour
    }).strip()

    # Step 3 → translate
    pet_story_es = (translate_prompt | llm | parser).invoke({
        "pet_story": pet_story
    }).strip()

    return {
        "pet_name": pet_name,
        "pet_story": pet_story,
        "pet_story_spanish": pet_story_es
    }


if __name__ == "__main__":
    result = generate_pet_info("unicorn", "green")
    print("Pet Name:", result["pet_name"])
    print("\nBackstory:", result["pet_story"])
    print("\nSpanish Translation:", result["pet_story_spanish"])
