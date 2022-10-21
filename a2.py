import pywhatkit as pwk

import datetime
now = datetime.datetime.now()

# print(now.hour)
# print(now.minute)
pwk.sendwhatmsg("+xxxxx","Hope this works NOW", now.hour , now.minute + 1)



