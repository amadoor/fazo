local band = bit.band
local ENTITIES = 5000
local FRAMES = 150
local IDLE, WANDER, COMBAT, FLEE = 1, 2, 3, 4

local seed = 777
local function random(n)
    seed = ((seed * 16838) % 32768 * 65536 + seed * 20077 + 12345) % 2147483648
    return seed % n + 1
end

local Entity = {}
Entity.__index = Entity

local function new_entity(index)
    return setmetatable({
        id = index,
        x = random(1000),
        y = random(1000),
        hp = random(100),
        state = random(4),
        timer = random(30),
        target = random(ENTITIES),
    }, Entity)
end

function Entity:distance_to(other)
    local dx = self.x - other.x
    local dy = self.y - other.y
    if dx < 0 then dx = -dx end
    if dy < 0 then dy = -dy end
    return dx + dy
end

function Entity:move(dx, dy)
    self.x = (self.x + dx) % 1000
    self.y = (self.y + dy) % 1000
end

function Entity:think(world, commands, frame)
    self.timer = self.timer - 1
    if self.state == IDLE then
        if self.timer <= 0 then
            self.state = WANDER
            self.timer = 10 + frame % 7
        end
    elseif self.state == WANDER then
        self:move(frame % 3 - 1, (frame + self.id) % 3 - 1)
        if self.timer <= 0 then self.state = IDLE self.timer = 5 end
        local foe = world[self.target]
        if self:distance_to(foe) < 40 then
            self.state = COMBAT
            self.timer = 4
        end
    elseif self.state == COMBAT then
        local foe = world[self.target]
        foe.hp = foe.hp - (self.id % 5 + 1)
        commands[#commands + 1] = { kind = "hit", source = self.id, target = foe.id }
        if foe.hp <= 0 then
            commands[#commands + 1] = { kind = "down", target = foe.id }
            foe.hp = 100
        end
        if self.hp < 30 then self.state = FLEE end
        if self.timer <= 0 then self.state = WANDER self.timer = 8 end
    else
        self:move(self.id % 3 - 1, -1)
        self.hp = self.hp + 1
        if self.hp > 60 then self.state = IDLE self.timer = 6 end
    end
    if self.hp > 100 then self.hp = 100 end
end

local world = {}
for i = 1, ENTITIES do
    world[i] = new_entity(i)
end

local commands = {}
for frame = 1, FRAMES do
    for i = 1, ENTITIES do
        world[i]:think(world, commands, frame)
    end
    if #commands > 4000 then
        commands = {}
    end
end

local checksum = 0
for i = 1, ENTITIES do
    local e = world[i]
    checksum = band(checksum * 31 + e.x + e.y * 7 + e.hp * 13 + e.state, 0x7fffffff)
end
print(#commands, checksum)
