
let user = {
	id: 0,
	name: "AiVA Bot",
	number: "+91 91231 40293",
	pic: "https://img.favpng.com/0/5/8/computer-icons-internet-bot-robot-clip-art-png-favpng-Rdyq3bYFvVSezi2a6wpXXGKfV.jpg"
};

let contactList = [
	{
		id: 0,
		name: "AiVA Bot",
		number: "+91 91231 40293",
		pic: "",
		lastSeen: "Apr 29 2018 17:58:02"
	},
	{
		id: 1,
		name: "Person 1",
		number: "+91 98232 37261",
		pic: "static/assets/images/prof.jpeg",
		lastSeen: "Apr 28 2018 22:18:21"
	},
	{
		id: 2,
		name: "Person 2",
		number: "+91 72631 2937",
		pic: "static/assets/images/prof.jpeg",
		lastSeen: "Apr 28 2018 19:23:16"
	},
	{
		id: 3,
		name: "Person 3",
		number: "+91 98232 63547",
		pic: "static/assets/images/prof.jpeg",
		lastSeen: "Apr 29 2018 11:16:42"
	},
	{
		id: 4,
		name: "Person 4",
		number: "+91 72781 38213",
		pic: "static/assets/images/prof.jpeg",
		lastSeen: "Apr 27 2018 17:28:10"
	}
];

let groupList = [
	{
		id: 1,
		name: "Person 1",
		members: [0, 1, 3],
		pic: "static/assets/images/prof.jpeg"
	},
	{
		id: 2,
		name: "Person 2",
		members: [0, 2],
		pic: "static/assets/images/prof.jpeg"
	},
	{
		id: 3,
		name: "Person 3",
		members: [0],
		pic: "static/assets/images/prof.jpeg"
	}
];

// message status - 0:sent, 1:delivered, 2:read

let messages = [
	{
		id: 0,
		sender: 2,
		body: "",
		time: "April 25, 2018 13:21:03",
		status: 2,
		recvId: 0,
		recvIsGroup: false
	},
	{
		id: 1,
		sender: 0,
		body: "",
		time: "April 25, 2018 13:22:03",
		status: 2,
		recvId: 2,
		recvIsGroup: false
	},
	{
		id: 2,
		sender: 0,
		body: "",
		time: "April 25, 2018 18:15:23",
		status: 2,
		recvId: 3,
		recvIsGroup: false
	},
	{
		id: 3,
		sender: 3,
		body: "",
		time: "April 25, 2018 21:05:11",
		status: 2,
		recvId: 0,
		recvIsGroup: false
	},
	{
		id: 4,
		sender: 0,
		body: "",
		time: "April 26, 2018 09:17:03",
		status: 1,
		recvId: 3,
		recvIsGroup: false
	},
	{
		id: 5,
		sender: 3,
		body: "",
		time: "April 27, 2018 18:20:11",
		status: 0,
		recvId: 1,
		recvIsGroup: true
	},
	{
		id: 6,
		sender: 1,
		body: "",
		time: "April 27, 2018 17:23:01",
		status: 1,
		recvId: 0,
		recvIsGroup: false
	},
	{
		id: 7,
		sender: 0,
		body: "",
		time: "April 27, 2018 08:11:21",
		status: 2,
		recvId: 2,
		recvIsGroup: false
	},
	{
		id: 8,
		sender: 2,
		body: "",
		time: "April 27, 2018 08:22:12",
		status: 2,
		recvId: 0,
		recvIsGroup: false
	},
	{
		id: 9,
		sender: 0,
		body: "",
		time: "April 27, 2018 08:31:23",
		status: 1,
		recvId: 2,
		recvIsGroup: false
	},
	{
		id: 10,
		sender: 0,
		body: "",
		time: "April 27, 2018 22:41:55",
		status: 2,
		recvId: 4,
		recvIsGroup: false
	},
	{
		id: 11,
		sender: 1,
		body: "",
		time: "April 28 2018 17:10:21",
		status: 0,
		recvId: 1,
		recvIsGroup: true
	}
];

let MessageUtils = {
	getByGroupId: (groupId) => {
		return messages.filter(msg => msg.recvIsGroup && msg.recvId === groupId);
	},
	getByContactId: (contactId) => {
		return messages.filter(msg => {
			return !msg.recvIsGroup && ((msg.sender === user.id && msg.recvId === contactId) || (msg.sender === contactId && msg.recvId === user.id));
		});
	},
	getMessages: () => {
		return messages;
	},
	changeStatusById: (options) => {
		messages = messages.map((msg) => {
			if (options.isGroup) {
				if (msg.recvIsGroup && msg.recvId === options.id) msg.status = 2;
			} else {
				if (!msg.recvIsGroup && msg.sender === options.id && msg.recvId === user.id) msg.status = 2;
			}
			return msg;
		});
	},
	addMessage: (msg) => {
		msg.id = messages.length + 1;
		messages.push(msg);
	}
};