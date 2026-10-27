#include "tutor.hpp"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <nlohmann/json.hpp>

namespace parrot {

namespace {

std::string level_guide(const std::string& level) {
    // Each level with an example of its style: small models follow an example better than a description.
    if (level == "A2")
        return "A2 (elementary): short sentences of 5 to 9 words, the 1,500 most common words, present and past simple, "
               "no idioms. One question at a time. For example: \"I like pizza too. What do you eat for lunch?\"";
    if (level == "B1")
        return "B1 (intermediate): sentences of 8 to 14 words, everyday vocabulary, common tenses and modal verbs, "
               "few idioms. For example: \"That sounds like a fun day out. Have you been to that cinema before, or "
               "was it your first time?\"";
    if (level == "B2")
        return "B2 (upper intermediate): natural sentences of 12 to 20 words, wider vocabulary, conditionals, the "
               "passive voice, common idioms and phrasal verbs. For example: \"I can see why you'd be torn; if I were in "
               "your shoes, I'd probably weigh up the commute before deciding.\"";
    return "C1 (advanced): long, natural sentences of 15 to 30 words, as between educated adults: nuanced vocabulary, "
           "idioms, any structure. For example: \"It's fascinating how a single trip can reshape your priorities, "
           "though I suspect the novelty wears off once everyday routines reassert themselves.\"";
}

// Lowercase, letters, digits and apostrophes, single spaces: how "original" is compared with what
// was said (the transcript's case and punctuation are the recognizer's, not the learner's).
std::string norm(const std::string& s) {
    std::string out;
    bool space = true;
    for (unsigned char c : s) {
        if (std::isalnum(c) || c == '\'') {
            out += static_cast<char>(std::tolower(c));
            space = false;
        } else if (!space) {
            out += ' ';
            space = true;
        }
    }
    while (!out.empty() && out.back() == ' ') out.pop_back();
    return out;
}

bool contains_words(const std::string& hay, const std::string& needle) {
    std::string h = " " + norm(hay) + " ", n = " " + norm(needle) + " ";
    return n.size() > 2 && h.find(n) != std::string::npos;
}

std::vector<std::string> split_words(const std::string& s) {
    std::vector<std::string> out;
    std::string w;
    for (char c : s) {
        if (std::isspace(static_cast<unsigned char>(c))) {
            if (!w.empty()) out.push_back(w), w.clear();
        } else {
            w += c;
        }
    }
    if (!w.empty()) out.push_back(w);
    return out;
}

std::string join(const std::vector<std::string>& w, size_t a, size_t b) {
    std::string out;
    for (size_t i = a; i < b; ++i) out += (out.empty() ? "" : " ") + w[i];
    return out;
}

}  // namespace

std::vector<Correction> split_correction(const Correction& c, int context) {
    auto a = split_words(c.original), b = split_words(c.corrected);
    std::vector<std::string> na, nb;
    for (auto& w : a) na.push_back(norm(w));
    for (auto& w : b) nb.push_back(norm(w));
    // Longest common subsequence of the words; the rest are the changes.
    size_t n = a.size(), m = b.size();
    std::vector<std::vector<int>> L(n + 1, std::vector<int>(m + 1, 0));
    for (size_t i = n; i-- > 0;)
        for (size_t j = m; j-- > 0;) L[i][j] = na[i] == nb[j] ? L[i + 1][j + 1] + 1 : std::max(L[i + 1][j], L[i][j + 1]);
    struct Hunk { size_t a0, a1, b0, b1; };
    std::vector<Hunk> hunks;
    bool trimmed = false;
    size_t i = 0, j = 0;
    while (i < n || j < m) {
        if (i < n && j < m && na[i] == nb[j]) { ++i, ++j; continue; }
        Hunk h{i, i, j, j};
        while ((i < n || j < m) && !(i < n && j < m && na[i] == nb[j])) {
            if (j >= m || (i < n && L[i + 1][j] >= L[i][j + 1])) ++i; else ++j;
        }
        h.a1 = i, h.b1 = j;
        if (h.a0 == h.a1 && (h.a0 == 0 || h.a0 == n)) {
            trimmed = true;
            continue;
        }
        // Changes a few words apart are one mistake.
        if (!hunks.empty() && h.a0 - hunks.back().a1 <= static_cast<size_t>(context)) {
            hunks.back().a1 = h.a1, hunks.back().b1 = h.b1;
        } else {
            hunks.push_back(h);
        }
    }
    if (hunks.empty()) return {};
    if (hunks.size() == 1 && n <= 6 && !trimmed) return {c};  // already short
    std::vector<Correction> out;
    for (const auto& h : hunks) {
        // Context around the change: what comes before and after, the same in both.
        size_t before = std::min<size_t>(context, std::min(h.a0, h.b0));
        size_t after = std::min<size_t>(context, std::min(n - h.a1, m - h.b1));
        Correction x{join(a, h.a0 - before, h.a1 + after), join(b, h.b0 - before, h.b1 + after), "", c.kind};
        // The explanation goes with the change it talks about (or with the first, if none is named).
        bool named = false;
        for (size_t k = h.a0; k < h.a1 && !named; ++k) named = contains_words(c.explanation, a[k]);
        for (size_t k = h.b0; k < h.b1 && !named; ++k) named = contains_words(c.explanation, b[k]);
        if (named) x.explanation = c.explanation;
        out.push_back(x);
    }
    if (std::none_of(out.begin(), out.end(), [](const Correction& x) { return !x.explanation.empty(); }))
        out.front().explanation = c.explanation;
    return out;
}

std::string tutor_system(const std::string& level) {
    return "You are Parrot, a friendly conversation partner for someone learning English. You talk with them about "
           "everyday things, like a curious friend, not like a teacher.\n\n"
           "Speak at their level: " + level_guide(level) + "\n\n"
           "Your turn has two parts:\n"
           "- reply: what you say next. React to what they said, add something of your own, and usually end with a "
           "question that keeps the conversation going. Never mention their mistakes in the reply; you may naturally "
           "reuse the correct form (if they said \"I goed\", you can say \"Oh, you went there?\"). At most three "
           "sentences.\n"
           "- corrections: mistakes in what they just said, at most three, the most important first. For each: "
           "original (only the wrong words and at most two words around them, copied exactly from what they said; never "
           "the whole sentence), corrected (those same words fixed), explanation (one short sentence, simple English), "
           "kind (grammar for forms and word order, vocabulary for a wrong or invented word, word_choice for a real word "
           "that does not fit). One correction per mistake. Leave a correction empty "
           "(all fields \"\") when there is no further mistake. Do not correct capital letters or punctuation: you only "
           "hear them. Do not correct what is merely informal but natural. If they made no mistake, all three are empty.";
}

std::string tutor_schema() {
    nlohmann::json c = {
        {"type", "object"},
        {"properties",
         {{"original", {{"type", "string"}}},
          {"corrected", {{"type", "string"}}},
          {"explanation", {{"type", "string"}}},
          {"kind", {{"type", "string"}, {"enum", {"", "grammar", "vocabulary", "word_choice"}}}}}},
        {"required", {"original", "corrected", "explanation", "kind"}},
    };
    // Three fixed slots, not an array: small models loop on arrays or stop at the first item.
    nlohmann::json s = {
        {"type", "object"},
        {"properties",
         {{"reply", {{"type", "string"}}},
          {"corrections",
           {{"type", "object"},
            {"properties", {{"first", c}, {"second", c}, {"third", c}}},
            {"required", {"first", "second", "third"}}}}}},
        {"required", {"reply", "corrections"}},
    };
    return s.dump();
}

std::string tutor_user(const TutorRequest& r) {
    std::string u;
    if (!r.topic.empty()) u += "Topic of the conversation: " + r.topic + "\n\n";
    if (!r.history.empty()) {
        u += "The conversation so far:\n";
        // The last turns are enough, and keep the prompt short.
        size_t from = r.history.size() > 8 ? r.history.size() - 8 : 0;
        for (size_t i = from; i < r.history.size(); ++i)
            u += (r.history[i].role == "partner" ? "You: " : "Learner: ") + r.history[i].text + "\n";
        u += "\n";
    }
    if (r.said.empty()) u += "Start the conversation with a greeting and a first question about the topic.";
    else u += "The learner just said: \"" + r.said + "\"\n\nYour turn.";
    return u;
}

TutorReply tutor_parse(const std::string& json, const std::string& said) {
    TutorReply out;
    nlohmann::json j = nlohmann::json::parse(json);
    out.reply = j.value("reply", "");
    for (const char* slot : {"first", "second", "third"}) {
        if (!j["corrections"].contains(slot)) continue;
        const auto& c = j["corrections"][slot];
        Correction x{c.value("original", ""), c.value("corrected", ""), c.value("explanation", ""), c.value("kind", "")};
        if (x.kind.empty()) x.kind = "grammar";  // a real correction whose kind the model left out
        if (norm(x.original).empty() && norm(x.corrected).empty()) continue;
        if (said.empty()) {
            out.dropped.push_back("\"" + x.original + "\": nothing was said yet");
        } else if (!contains_words(said, x.original)) {
            out.dropped.push_back("\"" + x.original + "\": not what the learner said");
        } else if (norm(x.original) == norm(x.corrected)) {
            out.dropped.push_back("\"" + x.original + "\": the correction changes only case or punctuation");
        } else if (norm(x.corrected).empty()) {
            out.dropped.push_back("\"" + x.original + "\": no corrected words");
        } else if (norm(x.original).size() > 2 * norm(x.corrected).size() && contains_words(x.original, x.corrected)) {
            // "We arrived at the airport early." -> "arrived": a quote of the sentence, not a fix.
            out.dropped.push_back("\"" + x.original + "\": the correction only quotes part of it");
        } else if (std::any_of(out.corrections.begin(), out.corrections.end(),
                               [&](const Correction& y) { return norm(y.original) == norm(x.original); })) {
            continue;
        } else {
            for (auto& part : split_correction(x)) {
                if (norm(part.original) == norm(part.corrected)) continue;
                if (std::any_of(out.corrections.begin(), out.corrections.end(),
                                [&](const Correction& y) { return norm(y.original) == norm(part.original); }))
                    continue;
                out.corrections.push_back(part);
            }
        }
    }
    return out;
}

TutorReply tutor_turn(Llm& llm, const TutorRequest& r) {
    auto t0 = std::chrono::steady_clock::now();
    std::string raw = llm.complete(tutor_system(r.level), tutor_user(r), tutor_schema(), 512);
    TutorReply out = tutor_parse(raw, r.said);
    out.ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    return out;
}

}  // namespace parrot
