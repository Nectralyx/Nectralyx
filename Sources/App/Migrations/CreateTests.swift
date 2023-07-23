//
//  CreateTests.swift
//  
//
//  Created by Morgan Keay on 2023-07-21.
//

import Fluent

struct CreateTests: Migration {
    func prepare(on database: FluentKit.Database) -> EventLoopFuture<Void> {
        return database.schema("tests")
            .id()
            .field("title", .string, .required)
            .create()
    }
    
    func revert(on database: FluentKit.Database) -> EventLoopFuture<Void> {
        return database.schema("tests").delete()
    }
    
    
}
