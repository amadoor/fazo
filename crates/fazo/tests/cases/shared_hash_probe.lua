local keys = {}
for i = 1, 64 do keys[i] = "k" .. i end
local long = string.rep("long key ", 8)

local function bump(t, k, by)
    t[k] = (t[k] or 0) + by
end

local function bump_calling(t, k, f)
    t[k] = (t[k] or 0) + f()
end

local function dump(t)
    local names = {}
    for k, v in pairs(t) do names[#names + 1] = tostring(k) .. "=" .. tostring(v) end
    table.sort(names)
    return table.concat(names, " ")
end

local plain = {}
local grown = {}
local seen = {}
local proxy = setmetatable({}, {
    __index = function(_, k) seen[#seen + 1] = "get " .. tostring(k); return 10 end,
    __newindex = function(t, k, v) seen[#seen + 1] = "set " .. tostring(k); rawset(t, k, v) end,
})
local defaults = setmetatable({}, { __index = { k1 = 100, k2 = 200 } })

for round = 1, 3000 do
    local k = keys[round % 64 + 1]
    bump(plain, k, 1)
    bump(plain, round % 5, 1)
    bump(plain, (round % 3) + 0.5, 1)
    bump(plain, round % 2 == 0, 1)
    bump(plain, long, 1)
    if round % 7 == 0 then plain[k] = nil end
    bump_calling(grown, k, function()
        for i = 1, round % 40 do grown["g" .. round .. "_" .. i] = i end
        return 1
    end)
    bump_calling(grown, keys[(round * 7) % 64 + 1], function()
        grown[k] = nil
        return 2
    end)
    if round % 500 == 0 then
        bump(proxy, k, 1)
        bump(proxy, "fresh" .. round, 1)
    end
    bump(defaults, keys[round % 4 + 1], 1)
end

print(dump(plain))
local count, total = 0, 0
for _, v in pairs(grown) do count = count + 1; total = total + v end
print(count, total)
print(dump(proxy))
print(table.concat(seen, ","))
print(dump(defaults))
