#Zehaan Walji
#Dec 11th, 2024
#Twilio API



#Imports
import os
from twilio.rest import Client
from langchain.tools import tool




@tool("Send Emergency Notification")
def send_emergency_notification(message):
    """Send an emergency notification via SMS or call"""
    account_sid = os.getenv('TWILIO_ACCOUNT_SID')
    auth_token = os.getenv('TWILIO_AUTH_TOKEN')

    if not account_sid or not auth_token:
        return "Twilio credentials are missing. Please set TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN."

    client = Client(account_sid, auth_token)

    try:
        notification = client.messages.create(
            body=message,
            from_='+13613014372',  # Twilio number
            to='+14038190081'      # Your test phone number
        )
        return f"Emergency notification sent successfully: {notification.sid}"
    except Exception as e:
        return f"Failed to send emergency notification: {str(e)}"

class RiskEscalationTools:
    # Use the standalone tool function
    notify_emergency_contacts = send_emergency_notification
