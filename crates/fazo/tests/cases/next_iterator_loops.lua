local function walk(t)
    local keys = {}
    for k, v in pairs(t) do keys[#keys + 1] = tostring(k) .. ":" .. tostring(v) end
    table.sort(keys)
    return table.concat(keys, " ")
end

local mixed = { 10, 20, 30, x = 1, y = 2, [2.5] = "f", [true] = "t" }
mixed[100] = "far"
local rounds = {}
for round = 1, 400 do
    local sum = 0
    for k, v in next, mixed do
        if type(v) == "number" then sum = sum + v end
        if type(k) == "string" then mixed[k] = v end
    end
    rounds[#rounds + 1] = sum
end
print(rounds[1], rounds[400], walk(mixed))

local clearing = {}
for i = 1, 200 do clearing["k" .. i] = i; clearing[i] = i end
local seen, total = 0, 0
for k, v in pairs(clearing) do
    clearing[k] = nil
    seen = seen + 1
    total = total + v
end
print(seen, total, next(clearing))

local only_key = 0
for k in pairs({ a = 1, b = 2, c = 3, 4, 5 }) do only_key = only_key + #tostring(k) end
print(only_key)

local three = 0
for k, v, extra in pairs({ 7, 8, 9 }) do
    three = three + k * v
    assert(extra == nil)
end
print(three)

local proxied = setmetatable({}, { __pairs = function(t) return function(_, k) if (k or 0) < 3 then return (k or 0) + 1, "p" end end, t, nil end })
local count = 0
for k, v in pairs(proxied) do count = count + k; assert(v == "p") end
print(count)

local bad_ok, bad = pcall(function()
    for _ in next, { 1 }, "missing" do end
end)
print(bad_ok, (bad:gsub("^.-:%d+: ", "")))

local not_table_ok, not_table = pcall(function()
    for _ in next, 42 do end
end)
print(not_table_ok, (not_table:gsub("^.-:%d+: ", "")))

local swapped = { a = 1, b = 2, c = 3 }
local iterator = next
local swaps = 0
for round = 1, 300 do
    if round == 150 then iterator = function(t, k) return next(t, k) end end
    for _ in iterator, swapped do swaps = swaps + 1 end
end
print(swaps)

local list = {}
for i = 1, 50 do list[i] = i * 3 end
list[30] = nil
local fallback = setmetatable({ 1, 2 }, { __index = function(_, i) if i < 6 then return i * 10 end end })
local through = setmetatable({}, { __index = { 5, 6, 7 } })
local ipairs_total = 0
for round = 1, 300 do
    for i, v in ipairs(list) do ipairs_total = ipairs_total + i + v end
    for i, v in ipairs(fallback) do ipairs_total = ipairs_total + i * v end
    for i, v in ipairs(through) do ipairs_total = ipairs_total + v end
    if round % 100 == 0 then
        for i in ipairs("abc") do ipairs_total = ipairs_total + i end
    end
end
print(ipairs_total)

local hinted = {}
for i = 1, 64 do hinted["h" .. i] = i end
for i = 1, 64, 3 do hinted["h" .. i] = nil end
collectgarbage()
for i = 1, 64, 6 do hinted["h" .. i] = -i end
local seen, visits = 0, 0
for round = 1, 40 do
    for k, v in pairs(hinted) do
        visits = visits + 1
        seen = seen + v
        if round % 7 == 0 and v % 5 == 0 then hinted[k] = nil end
    end
end
local jumps = 0
for k in pairs(hinted) do
    local after = next(hinted, k)
    if after ~= nil then jumps = jumps + 1 end
    next(hinted, "h2")
end
print(seen, visits, jumps)
