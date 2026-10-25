local t = {}
for i = 1, 12 do t["k" .. i] = i end
local sum = 0
for i = 1, 12 do sum = sum + t["k" .. i] end
print(sum)
t.k5 = nil
print(t.k5, t.k6, t["k12"])
t["k5"] = 50
local keys = {}
for k, v in pairs(t) do keys[#keys + 1] = k .. "=" .. v end
table.sort(keys)
print(table.concat(keys, " "))
