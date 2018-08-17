#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <string.h>
#include <ctype.h>
#include <time.h>
#include "constants.h"

#define BUFF_LEN 256

// structure to hold details of a booking
typedef struct {
  int id;
  char name[128];
  int num_seats;
  char timeslot[128];
} booking;

char buffer[BUFF_LEN];
booking *bookings[BUFF_LEN];
int bk_pos = 0;

// convert string to lower case
void str_to_lower(char *str) {
  for ( ; *str; ++str) *str = tolower(*str);
}

// read data (client's response) from socket
void read_response(int sock_fd) {
  int count;
  if ((count = recv(sock_fd, buffer, BUFF_LEN, 0)) < 0) {
    fprintf(stderr, S_READ_ERROR);
    exit(EXIT_FAILURE);
  }
  buffer[count] = '\0';
  printf("Client: %s\n", buffer);
}

// write data (response to client) to socket
void write_response(int sock_fd, char *response) {
  send(sock_fd, response, strlen(response), 0);
  printf("Bot: %s\n", response);
}

// check if booking details are valid
int validate_booking(booking *bking) {
  // validity check for time is ignored for obvious reasons
  return strlen(bking->name) && bking->num_seats && strlen(bking->timeslot);
}

// procedure for booking a table
void book_table(int sock_fd) {
  booking *new = malloc(sizeof(booking));
  new->id = rand() % 1000; // assign a random ID (reference number)

  // get details for booking a table
  write_response(sock_fd, S_BOT_QUES_NAME);
  read_response(sock_fd);
  strcpy(new->name, buffer);
  write_response(sock_fd, S_BOT_QUES_NUM_SEATS);
  read_response(sock_fd);
  new->num_seats = atoi(buffer);
  write_response(sock_fd, S_BOT_QUES_TIME);
  read_response(sock_fd);
  strcpy(new->timeslot, buffer);

  // check if details provided by the client are valid
  if (!validate_booking(new)) {
    write_response(sock_fd, S_BOT_INVALID_DETAILS_RESPONSE);
    read_response(sock_fd);
    str_to_lower(buffer);
    // check if user wants to re-enter the details
    if (strcmp(buffer, "yes")) {
      write_response(sock_fd, S_BOT_INVALID_RESPONSE " " S_BOT_BYE);
      return;
    }
    book_table(sock_fd); // re-ask all the details
    return;
  }

  // add the new booking
  bookings[bk_pos++] = new;
  sprintf(buffer, SF_BOT_BOOKING_SUCCESS " " S_BOT_BYE, new->id);
  write_response(sock_fd, buffer);
}

// search a booking by it's reference number (id)
int search(int id) {
  int i;
  for (i = 0; i < bk_pos; i++) {
    if (bookings[i]->id == id)
      return i; // found
  }
  return -1; // not found
}

// delete booking from array at index
void delete(int index) {
  int i = 0, j = 0;
  while (i < bk_pos) {
    if (i == index)
      free(bookings[i++]);
    else
      bookings[i++] = bookings[j++];
  }
  bk_pos--;
}

// procedure to cancel a booking
void cancel_table(int sock_fd) {
  int bk_index;
  write_response(sock_fd, S_BOT_QUES_REF_NUM);
  read_response(sock_fd);
  bk_index = search(atoi(buffer)); // get index for entered id (reference number)
  if (bk_index == -1) { // not found!
    write_response(sock_fd, S_BOT_INVALID_DETAILS_RESPONSE);
    read_response(sock_fd);
    str_to_lower(buffer);
    // check if user wants to re-enter cancellation details
    if (strcmp(buffer, "yes")) {
      write_response(sock_fd, S_BOT_INVALID_RESPONSE " " S_BOT_BYE);
      return;
    }
    cancel_table(sock_fd); // re-ask all the details
    return;
  }
  // confirm cancellation
  sprintf(buffer, SF_BOT_CONFIRM_CANCEL, bookings[bk_index]->name,
    bookings[bk_index]->num_seats, bookings[bk_index]->timeslot);
  write_response(sock_fd, buffer);
  read_response(sock_fd);
  str_to_lower(buffer);
  if (strcmp(buffer, "yes")) { // abort cancellation
    write_response(sock_fd, S_BOT_BOOKING_NOT_CANCELLED " " S_BOT_BYE);
    return;
  }
  // cancel the booking
  delete(bk_index);
  write_response(sock_fd, S_BOT_BOOKING_CANCELLED " " S_BOT_BYE);
}


// say hello to the client
void init_bot(int sock_fd) {
  int notok = 1;
  write_response(sock_fd, S_BOT_HELLO " " S_BOT_QUES_BOOK_CANCEL);
  do {
    read_response(sock_fd);
    str_to_lower(buffer);
    if (strstr(buffer, "cancel") != NULL) {
      notok = 0;
      cancel_table(sock_fd);
    } else if (strstr(buffer, "book") != NULL) {
      notok = 0;
      book_table(sock_fd);
    } else {
      write_response(sock_fd, S_BOT_INVALID_RESPONSE);
    }
  }  while (notok);
}

int main(int argc, char **argv) {
  int sock_fd, opt, port, sock_io;
  struct sockaddr_in address;
  int addr_len = sizeof(address);

  if (argc != 2) {
    fprintf(stderr, S_PARAMETER_ERROR SF_USAGE, argv[0]);
    return EXIT_FAILURE;
  }
  srand(time(NULL));
  port = atoi(argv[1]);
  // create a socket
  sock_fd = socket(AF_INET, SOCK_STREAM, 0);
  if (sock_fd < 0) {
    fprintf(stderr, S_SOCKET_ERROR);
    return EXIT_FAILURE;
  }
  // force socket & address re-use
  if (setsockopt(sock_fd, SOL_SOCKET, SO_REUSEADDR | SO_REUSEPORT, &opt, sizeof(opt))) {
    fprintf(stderr, S_REUSE_ERROR);
    return EXIT_FAILURE;
  }
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = INADDR_ANY;
  address.sin_port = htons(port);
  // Forcefully attach socket to the port
  if (bind(sock_fd, (struct sockaddr *) &address, sizeof(address)) < 0) {
    fprintf(stderr, SF_BIND_ERROR, port);
    return EXIT_FAILURE;
  }
  // listen for incoming connection
  if (listen(sock_fd, 10) < 0) {
    fprintf(stderr, S_LISTEN_ERROR);
    return EXIT_FAILURE;
  }
  // accept incoming connection
  while ((sock_io = accept(sock_fd, (struct sockaddr *) &address, (socklen_t*) &addr_len)) > -1) {
    printf("\nNew client:\n");
    init_bot(sock_io);
    close(sock_io);
    printf("Chat ended!\n");
  }

  close(sock_fd);
  return EXIT_SUCCESS;
}
