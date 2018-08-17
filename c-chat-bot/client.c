#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <string.h>

#define BUFF_LEN 128

char out_buffer[BUFF_LEN];
char in_buffer[BUFF_LEN] = "";

int main(int argc, char **argv) {
  struct sockaddr_in serv_addr;
  int sock_fd = 0, count;
  if (argc != 3) {
    printf("Usage: %s [address] [port]", argv[0]);
    return EXIT_FAILURE;
  }
  if ((sock_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    fprintf(stderr, "Unable to create socket\n");
    return EXIT_FAILURE;
  }
  memset(&serv_addr, '0', sizeof(serv_addr));
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(atoi(argv[2]));
  // Convert IPv4 and IPv6 addresses from text to binary form
  if(inet_pton(AF_INET, argv[1], &serv_addr.sin_addr) <= 0) {
    fprintf(stderr, "Invalid address!");
    return EXIT_FAILURE;
  }
  if (connect(sock_fd, (struct sockaddr *) &serv_addr, sizeof(serv_addr)) < 0) {
    fprintf(stderr, "Connection failed!\n");
    return EXIT_FAILURE;
  }
  // repeat the loop until server says bye
  do {
    count = recv(sock_fd, in_buffer, BUFF_LEN, 0);
    in_buffer[count] = '\0';
    printf("Bot: %s\n", in_buffer);
    if (strstr(in_buffer, "bye!") != NULL)
      break;
    printf("You: ");
    gets(out_buffer);
    send(sock_fd, out_buffer, strlen(out_buffer), 0);
  } while (1);

  close(sock_fd);
  return EXIT_SUCCESS;
}
