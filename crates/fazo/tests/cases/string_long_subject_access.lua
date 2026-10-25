local s = string.rep("abcdefgh", 4000)
local total = 0
for i = 1, #s do total = total + s:byte(i) end
print(total, #s, s:sub(-3), s:upper():sub(1, 4), s:lower():len(), s:reverse():sub(1, 3))
print(s:byte(1, 5))
print(s:byte(-2, -1))
print(string.byte(s, 32000, 32002))
print(utf8.len(s), utf8.codepoint(s, 1, 3))
local codes = 0
for _, code in utf8.codes(s) do codes = codes + code end
print(codes)
print(utf8.len("aé€😀"), utf8.offset("aé", 2), utf8.offset("aéb", -1))
print(string.len(12345), string.sub(123456, 2, 3), string.upper(1e3), string.rep(7, 3, "."))
print(pcall(utf8.codes, "\x80"))
print(pcall(string.byte, {}))
print(select("#", string.byte("")), string.sub("abc", 4), string.sub("abc", 2, 10))
