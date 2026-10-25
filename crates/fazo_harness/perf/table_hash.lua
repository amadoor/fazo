local t = {}
local keys = {}
for i = 1, 20000 do keys[i] = "key" .. i end
for round = 1, 10 do
    for i = 1, #keys do t[keys[i]] = (t[keys[i]] or 0) + round end
end
local sum = 0
for _, v in pairs(t) do sum = sum + v end
print(sum)
