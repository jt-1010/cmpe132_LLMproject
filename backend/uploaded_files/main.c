#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Function declarations */
int sentiment_score(char sentence[]);
char *sanitize_text(char sentence[]);

/* Main function */
int main() {

    // User input
    char sentence[1000];
    printf(""Enter a sentence: "");
    fgets(sentence, 1000, stdin);

    // Sanitize text
    char *clean_sentence = sanitize_text(sentence);

    // Get sentiment score
    int score = sentiment_score(clean_sentence);
    
    // Output sentiment score
    printf(""Sentiment score: %d\n"", score);

    return 0;
}

/* Function to calculate sentiment score */
int sentiment_score(char sentence[]) {

    int score = 0;
    char *word = strtok(sentence, "" "");

    while (word != NULL) {

        if(strcmp(word, ""good"") == 0 || strcmp(word, ""excellent"") == 0 || strcmp(word, ""great"") == 0){
            score += 2;
        }
        else if(strcmp(word, ""bad"") == 0 || strcmp(word, ""terrible"") == 0 || strcmp(word, ""awful"") == 0){
            score -= 2;
        }
        else{
            score += 1;
        }

        word = strtok(NULL, "" "");
    }

    return score;
}

/* Function to sanitize text */
char *sanitize_text(char sentence[]) {

    int length = strlen(sentence);
    char *new_sentence = (char*)malloc(length * sizeof(char));
    int i = 0, j = 0;

    while(sentence[i] != '\0'){

        //convert uppercase to lower case charachters
        if(sentence[i] >= 'A' && sentence[i] <= 'Z'){
            new_sentence[j] = sentence[i] + 32;
        }
        //copy numbers, lower case characters, and space characters
        else if((sentence[i] >= 'a' && sentence[i] <= 'z') || sentence[i] == ' ' || (sentence[i] >= '0' && sentence[i] <= '9')){
            new_sentence[j] = sentence[i];
        }

        i++;
        j++;
    }
    //append null character to the end of the new string
    new_sentence[j] = '\0';

    return new_sentence;
}"