CREATE TABLE instructor(
    inst_id INTEGER AUTO_INCREMENT UNIQUE,
    inst_name VARCHAR(20) NOT NULL,
    inst_email VARCHAR(20) NOT NULL,
    inst_password VARCHAR(20),
    games INTEGER[],
    player INTEGER[],
    def_game INTEGER,
    PRIMARY KEY (inst_id)
);

CREATE TABLE player(
    pl_id INTEGER AUTO_INCREMENT UNIQUE,
    pl_name VARCHAR(20) NOT NULL,
    pl_email VARCHAR(20) NOT NULL,
    pl_password VARCHAR(20),
    allowed_games INTEGER[],
    current_games INTEGER[],
    instructors INTEGER[],
    inventory INTEGER,
    backorder INTEGER,

    PRIMARY KEY (p_id)
);

CREATE TABLE game(
    g_id INTEGER AUTO_INCREMENT UNIQUE,
    g_length INTEGER,
    distr_present BOOLEAN,
    wholesaler_present BOOLEAN,
    inst_id INTEGER UNIQUE NOT NULL,
    pl_id INTEGER[] UNIQUE NOT NULL,
    backlog DOUBLE,
    holding DOUBLE,
    active BOOLEAN,
    info_sharing BOOLEAN,
    info_delay BOOLEAN,
    d_id INTEGER UNIQUE NOT NULL,
    rounds INTEGER,
    is_def_game BOOLEAN,
    player_weeks BOOLEAN[],
    FOREIGN KEY(inst_id) REFERENCES instructor(inst_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY(pl_id) REFERENCES player(pl_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY(d_id) REFERENCES demand_pattern(d_id) ON DELETE CASCADE ON UPDATE CASCADE,
    PRIMARY KEY (g_id)
);


CREATE TABLE demand_pattern(
    g_id INTEGER AUTO_INCREMENT UNIQUE,
    d_id INTEGER AUTO_INCREMENT UNIQUE,
    weeks INTEGER, 
    custom_demands INTEGER[],
    owned_by VARCHAR(20),
    related_games VARCHAR(20),
    game_name VARCHAR(2O),
    FOREIGN KEY(g_id) REFERENCES FROM game(g_id)
    PRIMARY KEY(d_id) 
    
);


