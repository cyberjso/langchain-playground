""""
A prompt style  where prompts are sent in batch to the model in a single request.

Useful in the following scenarios:
- Minimize latency since there will be less round trips between the consumer and the model provider
- Minmize token usage when dealing with multiple related prompts. The system prompt can be the same for all prompts in the batch

Avoid using this prompt style when:
- When prompts in the batch does not share the same context.
"""
from langchain_core.messages import SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from typing import Literal, List
from dotenv import load_dotenv

ALLOWED_AREAS = Literal["Maintenence", "Financial", "Sales support", "Logistics", "On Call"]

class ClassifiedMessage(BaseModel):
    original_question: str = Field(description = "he customer's full, original message text.")
    assigned_area: ALLOWED_AREAS = Field(description = "The single, best-fit company area based on the rules provided in the system prompt.")

class BatchClassification(BaseModel):
    classifications: List[ClassifiedMessage]

load_dotenv()

system_prompt =  """
You are a senior support analyst in a Telecom company. Your job is to read messages we receive via Website Contact section and foward them to the correct company area.
These are the possible areas you can send the customer to:
- Maintenence
- Financial
- Sales support
- Logistics
- On Call

Here there are a few examples:
question: ""My phone line has been completely dead since the storm this morning. I need it fixed right away."	
area: "Maintenence"

question: "I just received an email that says my service will be disconnected if I don't pay immediately. I need to sort this out."
area: "Financial"

question: "I want to upgrade my home internet speed. What packages do you have for 500 Mbps?"
area: "Sales support"

question: "The technician was supposed to be here between 1 PM and 5 PM today, but it’s 6 PM and no one has arrived."
area: "Logistics"

question: "My business server went down five minutes ago and all our phones are offline. This is an emergency."
area: "On Call"

When It is not possible to classify the message to any specific area, send to support.

You MUST return your entire output as a single, valid JSON object that adheres exactly to the following structure:
{format_instructions}

Analyze ALL {num_messages} messages and return the classification list.
"""

system_message =  SystemMessage(content = system_prompt)
user_prompt = "User Question: {question}"

prompt_template = ChatPromptTemplate.from_messages([system_message, ("user", user_prompt)])
model = ChatOpenAI(model = "gpt-4o", temperature = 0)

chain = prompt_template | model

questions = [
    {"question": "My phone line has been completely dead since the storm this morning. I need it fixed right away."},
    {"question": "I just received an email that says my service will be disconnected if I don't pay immediately. I need to sort this out."},
    {"question": "I want to upgrade my home internet speed. What packages do you have for 500 Mbps?"},
    {"question": "The technician was supposed to be here between 1 PM and 5 PM today, but it’s 6 PM and no one has arrived."},
    {"question": "My business server went down five minutes ago and all our phones are offline. This is an emergency."},
    {"question": "I've been trying to log into the billing portal for an hour, but it keeps giving me an error code."},
    {"question": "I need to confirm the date and time of my new fiber installation appointment."},
    {"question": "I can’t figure out how to add the international calling feature to my current plan."},
    {"question": "There's a constant, loud buzzing noise coming from the junction box outside my house."},
    {"question": "I have a sudden, total loss of service after 9 PM. Everything was fine earlier."},
    {"question": "Could I get an itemized breakdown of the charges on my last invoice? The total seems too high."},
    {"question": "I want to know if fiber optic is available at my new address before I move next month."},
    {"question": "The cable that runs from the pole to my house looks frayed and is touching a tree branch."},
    {"question": "I need to reschedule the technician visit that was set up for Friday."},
    {"question": "The emergency line is down, and this is a critical network failure."},
    {"question": "I'm still being charged for a premium channel subscription I cancelled three months ago."},
    {"question": "My Wi-Fi signal is cutting out every hour, making video calls impossible."},
    {"question": "I'm calling about a new quote for phone and internet service for a small office building."},
    {"question": "I received a notification that my modem shipment has been delayed again."},
    {"question": "The network monitoring system just alerted us to a major outage in the central region."},
    {"question": "I need to pause my service for six months while I travel abroad. What are my options?"},
    {"question": "My automatic payment was declined this month, and I need to set up a manual payment plan."},
    {"question": "We have an emergency. Our primary data center connection has failed completely."},
    {"question": "The installation crew left a big coil of wire exposed in my backyard. Can someone remove it?"},
    {"question": "I never received the tracking number for the new router I ordered last week."},
    {"question": "I’m looking for a better deal than what my current provider offers. Can you match or beat it?"},
    {"question": "There is visible sparking coming from our main equipment panel."},
    {"question": "I think I was double-charged for the same month's service. Can you check my statement?"},
    {"question": "I need to change the delivery address for the new modem."},
    {"question": "My satellite dish was knocked crooked during the high winds. I have no signal."},
    {"question": "We're experiencing intermittent service loss every few minutes."},
    {"question": "I want to negotiate the terms of a bulk service contract for 50 new homes."},
    {"question": "I need a copy of my last year’s payment history for tax purposes."},
    {"question": "My modem is constantly blinking red and won't connect to the network."},
    {"question": "I need immediate assistance; our connection is down, and we cannot operate the security system."},
    {"question": "The box with my equipment arrived today, but the power cord is missing."},
    {"question": "I would like to file a formal complaint about a charge on my credit card."},
    {"question": "Is there a discounted bundle package if I combine internet, TV, and mobile?"},
    {"question": "A truck just hit a pole down the street, and my service immediately stopped."},
    {"question": "I need clarification on the early termination fee mentioned in my contract."},
    {"question": "The self-installation kit instructions were confusing; I think I hooked up the lines incorrectly."},
    {"question": "I received a call offering me a discount, and I want to finalize that offer."},
    {"question": "I have not been able to track my new mobile phone shipment since yesterday."},
    {"question": "This is urgent: a major water main broke near our main service cable junction."},
    {"question": "I need to change my bank account details for the automatic monthly withdrawals."},
    {"question": "A rodent chewed through the exterior cable. I need an engineer to replace the line."},
    {"question": "I'm relocating and need to know which services I can transfer to my new state."},
    {"question": "The delivery date for my new service has changed three times now. What is the guaranteed date?"},
    {"question": "The whole neighborhood is currently without service. Is there an outage map I can check?"},
    {"question": "I need to cancel my service effective the end of the month."},
]

results = chain.batch(questions)

for result in results:
    print(f"Response:\n{result.content}\n")
    print(f"tokens used: {result.response_metadata['token_usage']}\n")