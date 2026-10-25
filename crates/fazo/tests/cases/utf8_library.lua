print(utf8.char(72, 228, 8364, 128512))
print(utf8.len("häll€"), utf8.len("\xff"), utf8.len("abc", 2))
print(utf8.codepoint("häl", 1, -1))
for p, c in utf8.codes("aé€") do print(p, c) end
print(utf8.offset("aé€", 3), utf8.offset("aé€", -1), utf8.offset("abc", 0, 2))
print(utf8.charpattern == "[\0-\x7F\xC2-\xFD][\x80-\xBF]*")
print(utf8.len("\u{D800}"), utf8.len("\u{D800}", 1, -1, true))
print(utf8.codepoint("\u{7FFFFFFF}", 1, 1, true))
