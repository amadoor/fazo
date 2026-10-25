local pieces = {"return ", "1 + ", "2"}
local i = 0
local f = load(function() i = i + 1; return pieces[i] end)
print(f())
print(load("x = "))
print(select(-1, "a", "b", "c"), select(2, "a", "b", "c"))
print(select('#'), select('#', nil, nil))
print(type(collectgarbage("count")), collectgarbage("isrunning"), collectgarbage())
print(collectgarbage("incremental"), collectgarbage("generational"), collectgarbage("incremental"))
print(load(function() return nil end) ~= nil)
