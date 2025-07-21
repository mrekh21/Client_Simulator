import random
import time
from typing import List, Dict, TypedDict, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from config import (
    LLM_MODEL_NAME, LLM_TEMPERATURE, COMPANY_INFO,
    SIMULATOR_EMAIL_ADDRESS, PLATFORM_EMAIL_ADDRESS,
    MIN_RESPONSE_DELAY_SECONDS, MAX_RESPONSE_DELAY_SECONDS,
    ADD_RANDOM_FOLLOW_UP_CHANCE, OPENAI_API_KEY, GOOGLE_API_KEY
)
from email_handler import send_email, fetch_unread_emails, get_gmail_service

# Initialize LLM with the loaded key
if "gpt" in LLM_MODEL_NAME:
    llm = ChatOpenAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, api_key=OPENAI_API_KEY)
elif "gemini" in LLM_MODEL_NAME:
    llm = ChatGoogleGenerativeAI(model=LLM_MODEL_NAME, temperature=LLM_TEMPERATURE, google_api_key=GOOGLE_API_KEY)
else:
    raise ValueError(f"Unsupported LLM_MODEL_NAME: {LLM_MODEL_NAME}")


# Define Agent State
class AgentState(TypedDict):
    chat_history: List[Dict[str, str]]  # [{'role': 'user', 'content': '...'}]
    current_request: Optional[Dict]  # {'people': 2, 'start_date': '...', 'end_date': '...', 'location': '...'}
    platform_draft: Optional[str]  # Last received tour draft from platform
    conversation_stage: str  # 'initial_request', 'info_gathering', 'draft_review', 'booking_confirmation', 'rejected'
    num_modifications_requested: int
    is_satisfied: bool
    is_booking_finalized: bool
    company_info: Dict  # Information about the company (country, name)
    last_email_sent_time: Optional[float]  # Timestamp when the last email was sent
    last_platform_email_subject: Optional[str]


class ExtractedInfo(BaseModel):
    """Extracted information from the client's request."""
    people: Optional[int] = Field(None, description="Number of people for the tour/booking.")
    start_date: Optional[str] = Field(None, description="Desired start date for the trip (e.g., 'YYYY-MM-DD').")
    end_date: Optional[str] = Field(None, description="Desired end date for the trip (e.g., 'YYYY-MM-DD').")
    location: Optional[str] = Field(None, description="Desired travel destination or location.")
    budget: Optional[str] = Field(None, description="Client's specified budget or price range.")
    preferences: Optional[str] = Field(None, description="Any specific preferences or requirements mentioned by the client (e.g., 'luxury', 'adventure', 'family-friendly').")

class EmailAnalysis(BaseModel):
    """Analysis of the received email and proposed action."""
    action: str = Field(..., description="The primary action the client agent should take. Examples: 'extract_info', 'respond_to_draft', 'confirm_booking', 'request_clarification', 'reject_proposal', 'initiate_conversation'.")
    extracted_info: Optional[ExtractedInfo] = Field(None, description="Structured information extracted from the email, if applicable.")
    missing_info: Optional[str] = Field(None, description="If 'action' is 'request_clarification' or 'extract_info', specifies what information is still missing.")
    response_type: Optional[str] = Field(None, description="The type of response expected from the client. Examples: 'acknowledgement', 'question', 'confirmation', 'rejection', 'information_update'.")
    content_summary: str = Field(..., description="A brief summary of the email's main content relevant to the conversation.")

class DraftEvaluation(BaseModel):
    """Evaluation of the platform's proposed tour draft."""
    decision: str = Field(..., description="Client's decision regarding the draft. Options: 'accept', 'request_modification', 'reject'.")
    reason: str = Field(..., description="The reason for the decision (e.g., 'budget too high', 'dates don't work', 'accepted as is').")
    message_to_platform: str = Field(..., description="The message that should be sent to the platform based on the decision.")


# --- LLM Chains (Prompts) ---

# Chain to analyze incoming email (initial request or platform response)
analyze_email_chain = (
        ChatPromptTemplate.from_messages([
            ("system", """You are a test client interacting with a travel platform.
        Analyze the content of the incoming email to determine its purpose and extract key information.

        **Here are the possible actions you should identify:**

        1.  **'initial_request'**: If this is a brand new inquiry from a potential client, asking to plan a trip.
            * Extract `people`, `start_date`, `end_date`, `location`, `budget`, `preferences`.
            * If information is missing for an 'initial_request', specify it in `missing_info` (e.g., "dates, location").

        2.  **'platform_question'**: If the platform is asking *you* (the client) for more information or clarification (e.g., "What are your preferred dates?", "What's your budget?").
            * Set `response_type` to 'question'.
            * Identify what specific `missing_info` the platform is asking for.

        3.  **'platform_response'**: If the platform is providing a general response, acknowledging, or providing general information that is *not* a question, draft, invoice, or rejection.
            * Set `response_type` to 'acknowledgement' or 'information'.

        4.  **'platform_draft'**: If the platform has sent a proposed tour itinerary or plan.
            * Set `response_type` to 'draft'.

        5.  **'platform_invoice'**: If the platform has sent a final invoice or booking confirmation.
            * Set `response_type` to 'invoice'.

        6.  **'platform_rejection'**: If the platform is rejecting the tour request or stating they cannot fulfill it.
            * Set `response_type` to 'rejection'.

        **Important Rules:**
        * **Never invent information if not explicitly provided.**
        * **Prioritize the most specific action.** For example, if it's a draft *with* a question, classify it primarily as 'platform_draft' and note the question in `content_summary`.
        * **Focus on the *new* content of the email**, not the quoted historical content.

        Provide the response in JSON format:
        ```json
        {{
            "action": "initial_request" | "platform_question" | "platform_response" | "platform_draft" | "platform_invoice" | "platform_rejection",
            "extracted_info": {{ // Only if 'initial_request' or if new info is provided in other actions
                "people": null,
                "start_date": null,
                "end_date": null,
                "location": null,
                "budget": null,
                "preferences": null
            }},
            "missing_info": "...", // What info *the platform* is asking for from the client (if action is 'platform_question') OR what client info is missing in 'initial_request'
            "response_type": "...", // e.g., "question", "draft", "invoice", "rejection", "acknowledgement", "information"
            "content_summary": "A brief summary of the email's main content."
        }}
        ```
        """
             ),
            ("human", "{email_content}")
        ])
        | llm.with_structured_output(EmailAnalysis)
)

# Chain to generate initial request or follow-up questions
generate_client_message_chain = (
        ChatPromptTemplate.from_messages([
            ("system", """You are simulating a client interacting with a travel platform.
         Your goal is to convey your needs clearly, respond to proposals, and finalize bookings.
         Maintain a polite and natural tone. Be concise but provide necessary details.

         Current Client Request: {current_request}
         Conversation History: {chat_history}
         Company Information: {company_info}

         Current Stage: {conversation_stage}
         Instruction: {instruction}
         """),
            ("human", "Generate the client's email/message based on the instruction.")
        ])
        | llm
)

# Chain to evaluate platform's tour draft
evaluate_draft_chain = (
        ChatPromptTemplate.from_messages([
            ("system", """You are a test client interacting with a travel platform.
        The platform has sent you a tour draft: {platform_draft}.
        Your original request was: {original_request}.
        Company Country: {company_country}.
        Number of modification requests so far: {num_modifications_requested}.

        Evaluate the draft according to the following rules:
        1. If the draft fully satisfies your request (number of people, dates, location in {company_country}, approximately desired activities)
           and you do not need any additional changes: confirm the tour.
        2. If the draft is not proposed for {company_country}, reject it and request a tour in the correct country.
        3. If something is missing (e.g., transport not specified, price missing), or you want 1-2 specific changes (e.g., a different hotel, a different activity, price reduction),
           but no more than 2 changes: request a modification.
        4. If you have already requested more than 2 changes and are still not satisfied: reject the tour.

        Return the result in JSON format:
        ```json
        {{
            "decision": "accept" | "request_modification" | "reject",
            "reason": "...",
            "message_to_platform": "..."
        }}
        ```
        """)
        ])
        | llm.with_structured_output(DraftEvaluation)
)


# --- Nodes ---

def initial_request_node(state: AgentState):
    """Generates the initial email to the platform."""
    print("\n--- Initial Request Node ---")
    instruction = (
        f"Start a tour request in {state['company_info']['country']} for {state['company_info']['name']} company. "
        "Specify the number of people, approximate dates, and why you are interested in this place. "
        "Be natural, sometimes slightly vague in the information to prompt the platform to ask questions."
    )

    sample_requests = [
        f"Hello! I want to plan a tour in {state['company_info']['country']} for 2 people, approximately mid-August for 10 days. I'm very interested in the nature and culture of {state['company_info']['country']}. Can you suggest anything?",
        f"Good morning! We are planning a trip to {state['company_info']['country']} with 3 friends. We want to plan 7-8 days in September. We are interested in wine tours and hiking. Do you have any options?",
        f"Greetings, I want to book a family tour in {state['company_info']['country']} (me, my spouse, and 2 children). Approximately end of July, for 5 days. We need a hotel with a pool and children's activities. Thank you in advance!"
    ]

    # Let LLM generate a more natural initial request
    llm_response = generate_client_message_chain.invoke({
        "chat_history": state["chat_history"],
        "current_request": state["current_request"],
        "company_info": state["company_info"],
        "instruction": instruction,
        "conversation_stage": state["conversation_stage"]
    }).content

    state['chat_history'].append({"role": "user", "content": llm_response})
    state['conversation_stage'] = 'initial_request'
    state['last_email_sent_time'] = time.time()  # Record send time
    return state


def send_email_node(state: AgentState, gmail_service):
    """Sends the last message from chat_history as an email."""
    print("\n--- Send Email Node ---")
    last_message = state['chat_history'][-1]['content']

    # Determine the base subject for the reply
    if state.get('last_platform_email_subject'):
        # If there's a subject from a previous platform email, use it as the base
        base_subject = state['last_platform_email_subject']
        # Ensure it starts with "Re:" if it's a reply and doesn't already have it
        if not base_subject.lower().startswith('re:'):
            subject = f"Re: {base_subject}"
        else:
            subject = base_subject # It's already a reply, keep it as is for threading
    else:
        # This is the very first email from the client, no previous subject to reply to
        subject = f"Tour Request to {state['company_info']['name']}"


    if state['conversation_stage'] == 'info_gathering':
        pass # Subject is already set correctly by base_subject logic
    elif state['conversation_stage'] == 'draft_review' and state['is_satisfied']:
        # If base_subject is "Re: Tour Draft" --> "Re: Tour Draft - CONFIRMATION"
        subject = f"{subject} - CONFIRMATION"
    elif state['conversation_stage'] == 'draft_review' and not state['is_satisfied']:
        subject = f"{subject} - Modification Request"
    elif state['conversation_stage'] == 'booking_confirmation':
        subject = f"{subject} - Final Confirmation"
    elif state['conversation_stage'] == 'rejected':
        subject = f"{subject} - Rejection"

    send_email(gmail_service, SIMULATOR_EMAIL_ADDRESS, PLATFORM_EMAIL_ADDRESS, subject, last_message)
    state['last_email_sent_time'] = time.time()
    return state


def receive_email_node(state: AgentState, gmail_service):
    """Fetches new emails from the platform."""
    print("\n--- Receive Email Node ---")
    # Implement delay to simulate real client behavior
    if state['last_email_sent_time']:
        elapsed_time = time.time() - state['last_email_sent_time']
        delay = random.randint(MIN_RESPONSE_DELAY_SECONDS, MAX_RESPONSE_DELAY_SECONDS)
        if elapsed_time < delay:
            print(f"Simulating client delay... Waiting for {delay - elapsed_time:.0f} seconds.")
            time.sleep(delay - elapsed_time)

    # Check for additional follow-up email from client (random chance)
    if random.random() < ADD_RANDOM_FOLLOW_UP_CHANCE and state['conversation_stage'] != 'initial_request' and not state[
        'is_booking_finalized']:
        follow_up_prompt = (
            "As a client, you might sometimes have additional thoughts. "
            "Create a short follow-up email (e.g., 'Sorry, I forgot to mention', 'I have another question')."
        )
        follow_up_message = generate_client_message_chain.invoke({
            "chat_history": state["chat_history"],
            "current_request": state["current_request"],
            "company_info": state["company_info"],
            "instruction": follow_up_prompt,
            "conversation_stage": state["conversation_stage"]
        }).content

        print(f"Sending random follow-up email: {follow_up_message}")
        send_email(gmail_service, SIMULATOR_EMAIL_ADDRESS, PLATFORM_EMAIL_ADDRESS, "Quick follow-up", follow_up_message)
        time.sleep(5)  # Small delay after follow-up

    new_emails = fetch_unread_emails(gmail_service)

    if not new_emails:
        print("No new emails from platform. Waiting...")
        return state

    # Take the most recent email from the platform
    # Assuming platform sends one response at a time for simplicity
    platform_email = new_emails[0]
    state['chat_history'].append({"role": "agent", "content": platform_email['body']})
    state['last_platform_email_subject'] = platform_email['subject']
    print(f"Received email from platform: {platform_email['subject']}")
    return state



def client_response_node(state: AgentState):
    """Generates client's response based on conversation stage."""
    print("\n--- Client Response Node ---")
    instruction = ""

    if state['conversation_stage'] == 'platform_response':
        instruction = "The platform has sent a general response or acknowledgement. Respond politely, acknowledging their message and indicating you are awaiting their next action (e.g., the tour draft)."
    elif state['conversation_stage'] == 'info_gathering':
        instruction = "The platform is asking for additional information. Answer its question. Sometimes add a small detail that the platform did not ask for."

    elif state['conversation_stage'] == 'draft_review':
        return state
    elif state['conversation_stage'] == 'booking_confirmation':
        instruction = "The platform has sent an invoice or final plan. Confirm its receipt and express satisfaction. Set is_booking_finalized to True."
        state['is_booking_finalized'] = True
    elif state['conversation_stage'] == 'rejected':
        instruction = "The platform rejected your tour offer. Write a short, regretful response. Set is_booking_finalized to True."
        state['is_booking_finalized'] = True

    if instruction: # Only generate if a specific instruction was set
        llm_response = generate_client_message_chain.invoke({
            "chat_history": state["chat_history"],
            "current_request": state["current_request"],
            "company_info": state["company_info"],
            "instruction": instruction,
            "conversation_stage": state["conversation_stage"]
        }).content
        state['chat_history'].append({"role": "user", "content": llm_response})
    else:
        # Fallback for unexpected scenarios where client_response_node is called without a clear instruction.
        print(f"WARNING: client_response_node called with unhandled conversation stage '{state['conversation_stage']}'. No new client message generated.")

    return state


def analyze_platform_response_node(state: AgentState):
    """Analyzes the platform's email and updates the state."""
    print("\n--- Analyze Platform Response Node ---")
    last_platform_response = state['chat_history'][-1]['content']
    analysis_result = analyze_email_chain.invoke({"email_content": last_platform_response})

    print(f"Platform response analysis: {analysis_result}")

    state['conversation_stage'] = analysis_result.action

    if analysis_result.action == 'initial_request':
        pass
    elif analysis_result.action == 'platform_question':
        if analysis_result.extracted_info:
             state['current_request'].update(analysis_result.extracted_info.model_dump())
        state['conversation_stage'] = 'info_gathering'
    elif analysis_result.action == 'platform_response' and analysis_result.response_type == 'draft':
        state['platform_draft'] = last_platform_response
        state['conversation_stage'] = 'draft_review'
    elif analysis_result.action == 'platform_invoice':
        state['conversation_stage'] = 'booking_confirmation'
    elif analysis_result.action == 'platform_rejection':
        state['conversation_stage'] = 'rejected'
        state['is_booking_finalized'] = True

    return state

def evaluate_draft_node(state: AgentState):
    """Evaluates the received draft and decides to accept, modify, or reject."""
    print("\n--- Evaluate Draft Node ---")
    if not state['platform_draft']:
        print("Error: No platform draft to evaluate.")
        state['chat_history'].append({"role": "user", "content": "Error: Expected a tour draft but received none."})
        state['conversation_stage'] = 'rejected'  # Or re-route for error handling
        return state

    # Pass relevant context to LLM for decision making
    original_request_str = str(state['current_request'])

    evaluation_result = evaluate_draft_chain.invoke({
        "platform_draft": state['platform_draft'],
        "original_request": original_request_str,
        "company_country": state['company_info']['country'],
        "num_modifications_requested": state['num_modifications_requested']
    })

    print(f"Draft evaluation result: {evaluation_result}")

    decision = evaluation_result.decision
    message_to_platform = evaluation_result.message_to_platform

    state['chat_history'].append({"role": "user", "content": message_to_platform})

    if decision == 'accept':
        state['is_satisfied'] = True
        state['conversation_stage'] = 'booking_confirmation'  # Expecting invoice next
    elif decision == 'request_modification':
        state['num_modifications_requested'] += 1
        state['is_satisfied'] = False
        state['conversation_stage'] = 'draft_review'  # Stay in review loop
    elif decision == 'reject':
        state['is_satisfied'] = False
        state['conversation_stage'] = 'rejected'
        state['is_booking_finalized'] = True  # Conversation ends here

    return state


# --- Conditional Edges ---

def decide_next_step(state: AgentState):
    """Determines the next step based on the conversation stage."""
    print(f"\n--- Deciding Next Step (Current Stage: {state['conversation_stage']}) ---")
    if state['is_booking_finalized']:
        print("Booking finalized. Ending conversation.")
        return END

    # After initial request is sent, client waits for platform's reply
    if state['conversation_stage'] == 'initial_request':
        return 'receive_email'

    # If the client just sent info or responded to a question/acknowledgment, it waits for platform's next move.
    elif state['conversation_stage'] == 'info_gathering': # Client just sent info (after platform_question)
        return 'receive_email'
    elif state['conversation_stage'] == 'platform_question': # Client just replied to platform's question
        return 'receive_email'
    elif state['conversation_stage'] == 'platform_response': # Client just replied to platform's general response/acknowledgement
        return 'receive_email' # Should wait for the actual draft or next step

    # If client is reviewing draft, it just sent a modification request or acceptance.
    elif state['conversation_stage'] == 'draft_review':
        # If client accepted or requested modifications, they wait for next platform action (invoice or new draft)
        return 'receive_email'

    # If booking confirmation email was just sent (e.g., client accepted invoice)
    elif state['conversation_stage'] == 'booking_confirmation':
        return END # Booking finalized, conversation ends here after client's final email
    # If rejection email was just sent
    elif state['conversation_stage'] == 'rejected':
        return END
    print(f"WARNING: decide_next_step: Unhandled conversation_stage: {state['conversation_stage']}. Defaulting to END.")
    return END # Fallback in case of unhandled stage


def route_platform_response(state: AgentState):
    """Routes based on the analysis of platform's response."""
    print("\n--- Routing Platform Response ---")
    last_platform_response_content = state['chat_history'][-1]['content']
    analysis_result = analyze_email_chain.invoke({"email_content": last_platform_response_content})

    # Store extracted info
    if analysis_result.extracted_info:
        state['current_request'].update(analysis_result.extracted_info.model_dump())
    state['platform_draft'] = None  # Reset draft
    state['is_satisfied'] = False  # Reset satisfaction

    # Based on the action from analyze_email_chain, decide the route (corrected in previous turn, confirmed correct here)
    if analysis_result.action == 'platform_question':
        return 'platform_question_received'
    elif analysis_result.action == 'platform_response' and analysis_result.response_type == 'draft':
        return 'platform_draft_received'
    elif analysis_result.action == 'platform_invoice':
        return 'platform_invoice_received'
    elif analysis_result.action == 'platform_rejection':
        return 'platform_rejection_received'
    elif analysis_result.action == 'platform_response':  # General acknowledgement/information from platform
        return 'platform_response_received'
    elif analysis_result.action == 'initial_request':
         print(f"WARNING: Platform response was incorrectly classified as 'initial_request'. Routing to generic client response.")
         return 'unknown_response'
    else:
        print(f"Unknown or unhandled platform response action: {analysis_result.action}, routing to generic client response.")
        return 'unknown_response'


# --- Build the Graph ---

def build_graph(gmail_service):
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("initial_request", lambda state: initial_request_node(state))
    workflow.add_node("send_email", lambda state: send_email_node(state, gmail_service))
    workflow.add_node("receive_email", lambda state: receive_email_node(state, gmail_service))
    workflow.add_node("analyze_platform_response", analyze_platform_response_node)
    workflow.add_node("client_response", client_response_node)  # For info gathering or final confirmations/rejections
    workflow.add_node("evaluate_draft", evaluate_draft_node)

    # Set entry point
    workflow.set_entry_point("initial_request")

    # Define edges and conditional edges
    workflow.add_edge("initial_request", "send_email")
    workflow.add_edge("send_email", "receive_email")  # After sending, wait for response

    # After receiving email, analyze it
    workflow.add_edge("receive_email", "analyze_platform_response")

    # Route based on platform response analysis
    workflow.add_conditional_edges(
        "analyze_platform_response",
        route_platform_response,
        {
            "platform_question_received": "client_response",
            "platform_draft_received": "evaluate_draft",
            "platform_invoice_received": "client_response",
            "platform_rejection_received": "client_response",  # Client responds to rejection
            "platform_response_received": "client_response",
            "unknown_response": "client_response"  # Generic response for unhandled cases
        }
    )

    # After client responds to a question/invoice/rejection
    workflow.add_edge("client_response", "send_email")

    # After evaluating draft, decide next action
    workflow.add_conditional_edges(
        "evaluate_draft",
        lambda state: "send_email" if not state["is_booking_finalized"] else END,
        {
            "send_email": "send_email",
            END: END
        }
    )

    # Final decision after sending email
    workflow.add_conditional_edges(
        "send_email",
        decide_next_step,
        {
            "receive_email": "receive_email",  # Continue loop
            END: END
        }
    )

    app = workflow.compile()
    return app

