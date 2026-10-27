// The conversation partner: a local language model that talks with you at your level (CEFR A2 to
// C1) and points out your mistakes discreetly: its reply carries on the conversation, and up to three
// corrections come separately, for the screen and for the review cards.
//
// The model answers in a JSON schema (a grammar constrains it). Its corrections are checked: the
// "original" must be words you actually said (an invented quote is dropped), the correction must
// change more than case or punctuation (the transcript has neither from your voice), and a
// correction identical to its original is no correction.
#pragma once

#include <string>
#include <vector>

#include "llm.hpp"

namespace parrot {

struct Turn {
    std::string role;  // "partner" or "learner"
    std::string text;
};

struct Correction {
    std::string original, corrected, explanation, kind;  // kind: grammar, vocabulary, word_choice
};

struct TutorReply {
    std::string reply;
    std::vector<Correction> corrections;
    std::vector<std::string> dropped;  // corrections the checks threw away, and why
    double ms = 0;
};

struct TutorRequest {
    std::string level = "B1";  // A2, B1, B2, C1
    std::string topic;
    std::vector<Turn> history;
    std::string said;  // what the learner just said (transcribed)
};

// The system prompt for a level (constant per level: the model keeps it computed).
std::string tutor_system(const std::string& level);
std::string tutor_schema();
// Builds the user message.
std::string tutor_user(const TutorRequest& r);
// Splits a correction of a long stretch ("Yesterday I goed ... it make me laugh") into one per change,
// each with up to `context` unchanged words around it ("I goed to" -> "I went to", "it make me" -> ...).
std::vector<Correction> split_correction(const Correction& c, int context = 2);
// Parses and checks the model's JSON.
TutorReply tutor_parse(const std::string& json, const std::string& said);

TutorReply tutor_turn(Llm& llm, const TutorRequest& r);

}  // namespace parrot
