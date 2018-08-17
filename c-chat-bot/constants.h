// S -> string, F -> formatted or not
#define SF_USAGE "Usage: %s [port]\n"
#define S_PARAMETER_ERROR "Invalid parameters!\n"
#define S_SOCKET_ERROR "Unable to create socket!\n"
#define S_REUSE_ERROR "Unable to re-use address/port.\n"
#define SF_BIND_ERROR "Unable to bind socket at port: %d\n"
#define S_LISTEN_ERROR "Unable to create listen socket!\n"
#define S_ACCEPT_ERROR "Unable to acceptyour incoming connection!\n"
#define S_READ_ERROR "Couldn't receive any response from the client. Bailing out...\n"
#define S_BOT_HELLO "Hello! I am the restaurant bot."
#define S_BOT_QUES_BOOK_CANCEL "Would you like to book a table or cancel a booked table?"
#define S_BOT_QUES_NAME "What's your name?"
#define S_BOT_QUES_NUM_SEATS "How many seats do you need?"
#define S_BOT_QUES_TIME "When can we expect you to be at the restaurant?"
#define S_BOT_QUES_REF_NUM "Tell me your reference number and I'll see what I can do!"
#define S_BOT_INVALID_RESPONSE "Sorry, I didn't get that. Try again!"
#define S_BOT_INVALID_DETAILS_RESPONSE "Invalid booking details provided! Do you want to try again?"
#define SF_BOT_BOOKING_SUCCESS "Great! We'll keep a table reserved for you.\nYour reference number is %d.\n"
#define S_BOT_BYE "bye!"
#define S_BOT_BOOKING_NOT_CANCELLED "Great! Booking not cancelled. See you soon!"
#define S_BOT_BOOKING_CANCELLED "Booking cancelled! We're sorry to hear that."
#define SF_BOT_CONFIRM_CANCEL "Name: %s, Seats: %d, Time: %s. Are you sure you want to cancel?"