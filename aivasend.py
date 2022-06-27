# number_list = ["whatsapp: +60177144551", "whatsapp: +60173234121", "whatsapp: +60125250692"]

import os
from twilio.rest import Client


# Find your Account SID and Auth Token at twilio.com/console
# and set the environment variables. See http://twil.io/secure

TWILIO_AUTH_TOKEN = "02a3986e7db30872b7a6006157d70bff"
TWILIO_ACCOUNT_SID = "ACb71e13c0bf1bd7b0e5f4e5a44e7a1d2f"

account_sid = TWILIO_ACCOUNT_SID
auth_token = TWILIO_AUTH_TOKEN
client = Client(account_sid, auth_token)

message = client.messages.create(
                              from_='whatsapp:+14155238886',
                              body='Hello there please talk to the bot by replying "join log-park"!',
                              to='whatsapp:+60177144551'
                          )




