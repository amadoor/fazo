local tables = {}
for i = 1, 300 do
  local t = {}
  for j = 1, 12 do t["row" .. i .. "_" .. j] = i * j end
  tables[i] = t
end
local total = 0
for i = 1, 300 do
  local t = tables[i]
  for j = 1, 12 do total = total + t["row" .. i .. "_" .. j] end
end
print(total)
local count = 0
for _ in pairs(tables[299]) do count = count + 1 end
print(count, tables[299]["row299_12"], tables[1]["row300_1"])
