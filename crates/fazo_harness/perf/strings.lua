local parts = {}
for i = 1, 9000 do
    parts[#parts + 1] = string.format("%d:%s", i, string.rep("x", i % 7))
end
local text = table.concat(parts, ",")
local count = 0
for word in text:gmatch("%d+") do count = count + 1 end
local replaced = text:gsub("x+", "y")
print(#text, count, #replaced, text:sub(1, 20), text:find("9999", 1, true))
