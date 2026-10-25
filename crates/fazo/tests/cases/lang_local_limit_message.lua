local lines = {}
for i = 1, 201 do lines[#lines + 1] = "local v" .. i .. " = " .. i end
print(select(2, load(table.concat(lines, "\n"))))
local ok = {}
for i = 1, 200 do ok[#ok + 1] = "local w" .. i .. " = " .. i end
print(load(table.concat(ok, "\n")) ~= nil)
