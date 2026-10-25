print(pcall(function() for _ in io.stdin:lines("x") do end end))
print(pcall(function() for _ in io.stdin:lines("l", "x") do end end))
print(io.stdin:lines("l") ~= nil)
