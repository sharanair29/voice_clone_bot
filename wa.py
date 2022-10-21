# Download the helper library from https://www.twilio.com/docs/python/install
import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure

TWILIO_ACCOUNT_SID = "ACb71e13c0bf1bd7b0e5f4e5a44e7a1d2f"
TWILIO_AUTH_TOKEN = "02a3986e7db30872b7a6006157d70bff"

def intro():
    account_sid = TWILIO_ACCOUNT_SID
    auth_token = TWILIO_AUTH_TOKEN
    client = Client(account_sid, auth_token)

    message = client.messages.create(
            body = "Hey start talking to the bot!",
            from_='whatsapp:+14155238886',
            to='whatsapp:+xxxx'
        )

intro()



