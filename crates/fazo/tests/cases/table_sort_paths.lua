local t = {}
for i = 1, 50 do t[i] = (i * 37) % 50 end
table.sort(t)
print(table.concat(t, " "))
local words = {}
for i = 1, 20 do words[i] = tostring((i * 7) % 20) end
table.sort(words)
print(table.concat(words, " "))
local mixed = {3, 1.5, 2, -1, 2.0}
table.sort(mixed)
print(table.concat(mixed, " "))
table.sort(mixed, function(a, b) return a > b end)
print(table.concat(mixed, " "))
print(pcall(table.sort, {1, "x", 2}))
print(pcall(table.sort, {1, true}))
print(pcall(table.sort, {"10", 9}))
print(pcall(table.sort, {3, 2, 1, 4, 5, 6, 7, 8, 9}, function(a, b) return true end))
local backing = {5, 3, 4}
local proxy = setmetatable({}, {__index = backing, __newindex = backing, __len = function() return #backing end})
table.sort(proxy)
print(backing[1], backing[2], backing[3], rawget(proxy, 1))
local V = {}
V.__lt = function(a, b) return a.v < b.v end
local items = {}
for i = 1, 5 do items[i] = setmetatable({v = (i * 3) % 5}, V) end
table.sort(items)
for i = 1, 5 do io.write(items[i].v, " ") end
print()
local strings = {"pear", "apple", "fig", "banana", "apple"}
table.sort(strings)
print(table.concat(strings, " "))
