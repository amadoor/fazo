local f = io.stdin
print(pcall(f.read, f, "x"))
print(pcall(f.read, f, "*z"))
print(pcall(f.seek, f, "bogus"))
print(pcall(f.setvbuf, f, "bogus"))
print(pcall(function() return f:read("x") end))
