FormAI_101087.c,VULNERABLE,"//FormAI DATASET v1.0 Category: Network Topology Mapper ; Style: multi-threaded
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <string.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <netdb.h>
#include <ifaddrs.h>

#define MAX_THREADS 10
#define MAX_IP_LENGTH 16

pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;
int active_threads = 0;

struct host_list {
    char hostname[MAX_IP_LENGTH];
    struct host_list *next;
};

void *scan_network(void *arg);
int check_ip(char *ip);

int main() {
    struct ifaddrs *ifaddr, *ifa;
    struct sockaddr_in *sock_addr;
    char *interface_name = ""eth0"";
    int family, s, err;

    if (getifaddrs(&ifaddr) == -1) {
        perror(""getifaddrs"");
        exit(EXIT_FAILURE);
    }

    for (ifa = ifaddr; ifa != NULL; ifa = ifa->ifa_next) {
        if (ifa->ifa_addr == NULL) {
            continue;
        }
        family = ifa->ifa_addr->sa_family;
        if (strcmp(ifa->ifa_name, interface_name) == 0 && family == AF_INET) {
            sock_addr = (struct sockaddr_in *)ifa->ifa_addr;
            break;
        }
    }

    freeifaddrs(ifaddr);

    char buf[MAX_IP_LENGTH];
    snprintf(buf, MAX_IP_LENGTH, ""%s"", inet_ntoa(sock_addr->sin_addr));

    struct in_addr ip;
    inet_pton(AF_INET, buf, &ip);

    printf(""Scanning network %s.0/24\n\n"", inet_ntoa(ip));

    pthread_t threads[MAX_THREADS];

    for (int i = 0; i < MAX_THREADS; i++) {
        char *ip_addr = (char *) malloc(sizeof(char) * MAX_IP_LENGTH);

        snprintf(ip_addr, MAX_IP_LENGTH, ""%s.%d"", inet_ntoa(ip), i + 1);

        err = pthread_create(&threads[i], NULL, scan_network, ip_addr);

        if (err) {
            fprintf(stderr, ""Error creating thread: %s\n"", strerror(err));
            exit(EXIT_FAILURE);
        }

        active_threads++;

        sleep(1);
    }

    for (int i = 0; i < MAX_THREADS; i++) {
        err = pthread_join(threads[i], NULL);

        if (err) {
            fprintf(stderr, ""Error joining thread: %s\n"", strerror(err));
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}

void *scan_network(void *arg) {
    char *ip_addr = (char *) arg;

    if (check_ip(ip_addr)) {
        pthread_mutex_lock(&mutex);
        printf(""%s is active\n"", ip_addr);
        pthread_mutex_unlock(&mutex);
    } else {
        pthread_mutex_lock(&mutex);
        printf(""%s is non-active\n"", ip_addr);
        pthread_mutex_unlock(&mutex);
    }

    free(ip_addr);
    active_threads--;

    pthread_exit(NULL);
}

int check_ip(char *ip) {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    struct sockaddr_in sock_addr;
    sock_addr.sin_family = AF_INET;
    inet_pton(AF_INET, ip, &sock_addr.sin_addr);
    sock_addr.sin_port = htons(80);

    int res = connect(sock, (struct sockaddr *)&sock_addr, sizeof(sock_addr));

    close(sock);
    return (res == 0) ? 1 : 0;
}",main,44.0,dereference failure: invalid pointer