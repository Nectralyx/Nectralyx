//
//  File.swift
//  
//
//  Created by Morgan Keay on 2023-07-21.
//

import Fluent
import Vapor

final class Test: Model, Content {
    static let schema = "tests"
    
    @ID(key: .id)
    var id: UUID?
    
    @Field(key: "title")
    var title: String
    
    init() { }
    
    init(id: UUID? = nil, title: String) {
        self.id = id
        self.title = title
    }
}
