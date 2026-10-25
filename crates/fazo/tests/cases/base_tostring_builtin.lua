local text = tostring(print)
print(text:sub(1, 10), text:find("builtin") == nil)
print(tostring(string.format):find("^function: ") ~= nil)
