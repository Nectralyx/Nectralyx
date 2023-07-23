//
//  TestController.swift
//  
//
//  Created by Morgan Keay on 2023-07-21.
//
import Vapor
import Fluent

struct TestController: RouteCollection {
    func boot(routes: Vapor.RoutesBuilder) throws {
        let test = routes.grouped("tests")
        test.get(use: index)
        test.post(use: create)
    }
    func index(req: Request) throws -> EventLoopFuture<[Test]> {
        return Test.query(on: req.db).all()
    }
    func create(req: Request) throws -> EventLoopFuture<HTTPStatus> {
        let test = try req.content.decode(Test.self)
        return test.save(on: req.db).transform(to: .ok)
    }
    
}
