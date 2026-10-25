local mode = arg[1] or "interpreted"
local RECORDS = 20000
local TOPICS = 200
local ROUNDS = 4
local OPERATORS = { "==", "~=", "<", "<=", ">", ">=" }
local SOURCES = { "speaker.vars", "player.factions", "player.stats", "player.items" }

local seed = 4242
local function random(n)
    seed = (seed * 1103515245 + 12345) % 2147483648
    return seed % n + 1
end

local function make_condition()
    local kind = random(4)
    local limit = kind == 3 and 100 or (kind == 1 and 10 or (kind == 2 and 10 or 6))
    return { kind = kind, index = random(16), op = random(6), value = random(limit) - 1 }
end

local records = {}
for i = 1, RECORDS do
    local conditions = {}
    for c = 1, random(6) do
        conditions[c] = make_condition()
    end
    records[i] = { id = i, topic = (i - 1) % TOPICS + 1, conditions = conditions }
end

local function test_interpreted(condition, speaker, player)
    local kind = condition.kind
    local left
    if kind == 1 then
        left = speaker.vars[condition.index]
    elseif kind == 2 then
        left = player.factions[condition.index]
    elseif kind == 3 then
        left = player.stats[condition.index]
    else
        left = player.items[condition.index]
    end
    local op = condition.op
    local value = condition.value
    if op == 1 then return left == value end
    if op == 2 then return left ~= value end
    if op == 3 then return left < value end
    if op == 4 then return left <= value end
    if op == 5 then return left > value end
    return left >= value
end

local function passes_interpreted(record, speaker, player)
    for _, condition in ipairs(record.conditions) do
        if not test_interpreted(condition, speaker, player) then
            return false
        end
    end
    return true
end

local function condition_source(condition)
    local left = SOURCES[condition.kind] .. "[" .. condition.index .. "]"
    return left .. " " .. OPERATORS[condition.op] .. " " .. condition.value
end

local function compile(record)
    local parts = {}
    for c, condition in ipairs(record.conditions) do
        parts[c] = condition_source(condition)
    end
    local source = "return function(speaker, player) return " .. table.concat(parts, " and ") .. " end"
    return load(source, "=dialogue" .. record.id)()
end

local by_topic = {}
for t = 1, TOPICS do
    by_topic[t] = {}
end
for i = 1, RECORDS do
    local bucket = by_topic[records[i].topic]
    bucket[#bucket + 1] = records[i]
end

local passes
if mode == "compiled" then
    for i = 1, RECORDS do
        records[i].passes = compile(records[i])
    end
    passes = function(record, speaker, player)
        return record.passes(speaker, player)
    end
else
    passes = passes_interpreted
end

local speaker = { vars = {} }
for v = 1, 16 do
    speaker.vars[v] = random(10) - 1
end
local player = { factions = {}, stats = {}, items = {} }
for f = 1, 16 do
    player.factions[f] = random(10) - 1
    player.stats[f] = random(100) - 1
    player.items[f] = random(6) - 1
end

local checksum = 0
local answered = 0
for round = 1, ROUNDS do
    for t = 1, TOPICS do
        local first = 0
        for _, record in ipairs(by_topic[t]) do
            if passes(record, speaker, player) then
                first = record.id
                break
            end
        end
        if first > 0 then answered = answered + 1 end
        checksum = (checksum * 31 + first) & 0x7fffffff
    end
    player.stats[round] = (player.stats[round] + 37) % 100
    speaker.vars[round] = (speaker.vars[round] + 3) % 10
end
print(mode, answered, checksum)
