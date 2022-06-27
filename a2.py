import pywhatkit as pwk

import datetime
now = datetime.datetime.now()

# print(now.hour)
# print(now.minute)
pwk.sendwhatmsg("+60125250692","Hope this works NOW", now.hour , now.minute + 1)
# pwk.sendwhatmsg("+60125250692","Hope this works", 1 , 4)
# pwk.sendwhatmsg_instantly("+60125250692","Testing")


