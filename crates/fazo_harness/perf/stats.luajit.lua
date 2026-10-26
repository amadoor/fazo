local floor = math.floor
local band = bit.band
local ACTORS = 2000
local FRAMES = 6
local SKILLS = 27

local seed = 12345
local function random(n)
    seed = ((seed * 16838) % 32768 * 65536 + seed * 20077 + 12345) % 2147483648
    return seed % n + 1
end

local function clamp(value, low, high)
    if value < low then return low end
    if value > high then return high end
    return value
end

local function new_actor(index)
    local skills = {}
    for s = 1, SKILLS do skills[s] = random(100) end
    return {
        id = index,
        level = random(30),
        strength = random(100),
        intelligence = random(100),
        willpower = random(100),
        agility = random(100),
        speed = random(100),
        endurance = random(100),
        personality = random(100),
        luck = random(100),
        skills = skills,
        fatigue_loss = random(40),
        carried = random(400),
        capacity = random(300) + 100,
        health = 0,
        magicka = 0,
        fatigue = 0,
        encumbrance = 0,
        speed_derived = 0,
    }
end

local function skill_value(actor, index, frame)
    local base = actor.skills[index]
    local attribute = actor.intelligence + actor.luck
    local modifier = (attribute + frame * index) % 21 - 10
    return clamp(base + modifier + floor(actor.level / 5), 0, 100)
end

local function derive(actor, frame)
    actor.health = clamp(actor.endurance * 2 + floor(actor.strength / 2) + actor.level, 1, 300)
    actor.magicka = clamp(actor.intelligence * 2 + floor(actor.willpower / 2), 0, 400)
    local fatigue_max = actor.strength + actor.willpower + actor.agility + actor.endurance
    actor.fatigue = clamp(fatigue_max - (actor.fatigue_loss + frame) % 40, 0, fatigue_max)
    actor.encumbrance = clamp(floor(actor.carried * 100 / actor.capacity), 0, 200)
    local burden = actor.encumbrance > 100 and floor((actor.encumbrance - 100) / 2) or 0
    actor.speed_derived = clamp(actor.speed - burden + floor(actor.agility / 10), 5, 150)
    local sum = 0
    for s = 1, SKILLS do
        sum = sum + skill_value(actor, s, frame)
    end
    return floor(sum / SKILLS)
end

local actors = {}
for i = 1, ACTORS do
    actors[i] = new_actor(i)
end

local checksum = 0
for frame = 1, FRAMES do
    for i = 1, ACTORS do
        local actor = actors[i]
        local average = derive(actor, frame)
        checksum = band(checksum * 31 + actor.health + actor.magicka + actor.fatigue + actor.speed_derived + average, 0x7fffffff)
    end
end
print(checksum)
