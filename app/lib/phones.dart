// The 39 English phonemes (ARPAbet, as the engine names them), in IPA and with an example word.

const phoneIpa = {
  'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð', //
  'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'ɡ', 'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
  'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ', 'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
  'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v', 'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ',
};

const phoneExample = {
  'AA': 'father', 'AE': 'cat', 'AH': 'cup', 'AO': 'thought', 'AW': 'now', 'AY': 'my', 'B': 'bed', 'CH': 'chair',
  'D': 'dog', 'DH': 'this', 'EH': 'bed', 'ER': 'bird', 'EY': 'day', 'F': 'fish', 'G': 'go', 'HH': 'hat', //
  'IH': 'sit', 'IY': 'see', 'JH': 'job', 'K': 'cat', 'L': 'leg', 'M': 'man', 'N': 'no', 'NG': 'sing',
  'OW': 'go', 'OY': 'boy', 'P': 'pen', 'R': 'red', 'S': 'sun', 'SH': 'she', 'T': 'tea', 'TH': 'think',
  'UH': 'book', 'UW': 'food', 'V': 'van', 'W': 'wet', 'Y': 'yes', 'Z': 'zoo', 'ZH': 'vision',
};

/// "θ as in think"
String describe(String phone) {
  final ipa = phoneIpa[phone];
  if (ipa == null) return phone;
  return '$ipa as in ${phoneExample[phone]}';
}
