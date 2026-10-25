print(string.gsub("abc", "%w", function() return false end))
print(string.gsub("abc", "%w", {a = false, b = "B"}))
print(string.gsub("abc", "%w", function(c) if c == "b" then return "x" end end))
